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
        self.mem_gpu = GPUMemory(tam_mem_gpu)
        self.q_bloques = Queue()




    def ejecutar_trabajo_en_gpu(self, kernel_tipo: int, vector1: list, vector2: list = None, radio_dif: int = 2, filas: int = None, columnas: int = None) -> list:

        self.mem_gpu.kernel.value = kernel_tipo

        self.mem_gpu.radio_dif.value = radio_dif
        
        tam_datos = len(vector1)
        self.mem_gpu.tam_datos.value = tam_datos

        # El Host copia los datos a la memoria de la GPU
        self.mem_gpu.dato1[:tam_datos] = vector1
        if vector2 is not None:
            self.mem_gpu.dato2[:tam_datos] = vector2

        if self.mem_gpu.kernel.value == DIFUMINAR_MAT :
            self.mem_gpu.filas.value = filas
            self.mem_gpu.columnas.value = columnas

        # Arrancamos los SMs (procesos)
        self.sms = [SM(self.nucleos_por_sm, self.mem_gpu, self.mem_gpu.tam_datos, self.q_bloques)
                    for _ in range(self.cant_sms)]
        for s in self.sms:
            s.start()

        # El ordenador divide el trabajo en bloques y los encola.
        for block_start in range(0, tam_datos, self.nucleos_por_sm):
            #Aqui usamos min para que en el último bloque no nos pasemos del tamaño de datos a procesar, 
            # que puede no ser del tamaño de CANT_NUCLEOS_POR_SM
            block_size = min(self.nucleos_por_sm, tam_datos - block_start)
            self.q_bloques.put((block_start, block_size))

        for _ in range(self.cant_sms):
            self.q_bloques.put(CENTINELA)
        for s in self.sms:
            s.join()

        # El Host coge el resultado de la memoria global y lo devuelve
        return self.mem_gpu.res[:tam_datos]

    def apagar(self):
        #Métodopar cerrar los sm al final del programa
        # Señal de finalización (centinela): ponemos un None por cada SM al final
        # para indicarles que terminen su bucle.
        pass