from multiprocessing import Queue

from kernels import DIFUMINAR_MAT
from sm import SM
from gpu_memory import GPUMemory

"""
PREGUNTAR SI LA CLASE GPU TIENE QUE SER TAMBIEN PROCESS 
"""
CENTINELA = None

class GPU:
    def __init__(self, cant_sms: int, nucleos_por_sm: int, tam_mem_gpu: int, tam_mem_sm: int):
        self.cant_sms = cant_sms
        self.nucleos_por_sm = nucleos_por_sm
        self.tam_mem_gpu = tam_mem_gpu
        self.tam_mem_sm = tam_mem_sm

    def ejecutar_trabajo_en_gpu(self, kernel_tipo: int, vector1: list, vector2: list = None, filas: int = None, columnas: int = None) -> list:

        mem_gpu = GPUMemory(self.tam_mem_gpu)
        q_bloques = Queue()

        mem_gpu.kernel.value = kernel_tipo
        
        tam_datos = len(vector1)
        mem_gpu.tam_datos.value = tam_datos

        # El Host copia los datos a la memoria de la GPU
        mem_gpu.dato1[:tam_datos] = vector1
        if vector2 is not None:
            mem_gpu.dato2[:tam_datos] = vector2

        if mem_gpu.kernel.value == DIFUMINAR_MAT :
            mem_gpu.filas.value = filas
            mem_gpu.columnas.value = columnas


        # Arrancamos los SMs (procesos)
        sms = [SM(self.nucleos_por_sm, mem_gpu, self.tam_mem_sm, q_bloques) for _ in range(self.cant_sms)]
        for s in sms:
            s.start()

        # El ordenador divide el trabajo en bloques y los encola.
        for block_start in range(0, tam_datos, self.nucleos_por_sm):
            #Aqui usamos min para que en el último bloque no nos pasemos del tamaño de datos a procesar, 
            # que puede no ser del tamaño de CANT_NUCLEOS_POR_SM
            block_size = min(self.nucleos_por_sm, tam_datos - block_start)
            q_bloques.put((block_start, block_size))


        # Señal de finalización (centinela): ponemos un None por cada SM al final
        # para indicarles que terminen su bucle.
        for _ in range(self.cant_sms):
            q_bloques.put(CENTINELA)

        # Espera a que todos los SMs acaben.
        for sm in sms:
            sm.join()

        # El Host coge el resultado de la memoria global y lo devuelve
        return mem_gpu.res[:tam_datos]