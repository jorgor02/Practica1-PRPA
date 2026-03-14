from multiprocessing import Queue
from kernels import DIFUMINAR_MAT
from sm import SM
from gpu_memory import GPUMemory

CENTINELA = None

class GPU:
    def __init__(self, cant_sms: int, nucleos_por_sm: int, tam_mem_gpu: int, tam_mem_sm: int):
        self.cant_sms = cant_sms
        self.nucleos_por_sm = nucleos_por_sm
        self.tam_mem_gpu = tam_mem_gpu
        self.tam_mem_sm = tam_mem_sm

        self.mem_gpu = GPUMemory(self.tam_mem_gpu)
        self.q_bloques = Queue()
        self.q_completados = Queue()

        # Arrancamos los SMs (procesos)
        self.sms = [SM(self.nucleos_por_sm, self.mem_gpu, self.tam_mem_sm, self.q_bloques, self.q_completados)
                     for _ in range(self.cant_sms)]
        for s in self.sms:
            s.start()

    def apagar(self):
        # El Host llamará a este método para detener todo al finalizar
        # Señal de finalización (centinela): ponemos un None por cada SM al final
        # para indicarles que terminen su bucle.
        for _ in range(self.cant_sms):
            self.q_bloques.put(CENTINELA)
        for sm in self.sms:
            sm.join()

    def ejecutar_trabajo_en_gpu(self, kernel_tipo: int, vector1: list, vector2: list | None = None, filas: int | None = None, columnas: int | None = None, radio: int = 1) -> list:
        
        self.mem_gpu.kernel.value = kernel_tipo
        tam_datos = len(vector1)
        self.mem_gpu.tam_datos.value = tam_datos
        self.mem_gpu.radio.value = radio

        # El Host copia los datos a la memoria de la GPU
        self.mem_gpu.dato1[:tam_datos] = vector1
        if vector2 is not None:
            self.mem_gpu.dato2[:tam_datos] = vector2
            
        if self.mem_gpu.kernel.value == DIFUMINAR_MAT :
            self.mem_gpu.filas.value = filas
            self.mem_gpu.columnas.value = columnas

        
        # El ordenador divide el trabajo en bloques y los encola.
        num_bloques = 0
        for block_start in range(0, tam_datos, self.nucleos_por_sm):
            #Aqui usamos min para que en el último bloque no nos pasemos del tamaño de datos a procesar, 
            # que puede no ser del tamaño de CANT_NUCLEOS_POR_SM
            block_size = min(self.nucleos_por_sm, tam_datos - block_start)
            self.q_bloques.put((block_start, block_size))
            num_bloques += 1


        # Ahora contamos cuántos bloques han respondido que han terminado, para saber
        # cuándo podemos coger el resultado de la memoria global.
        for _ in range(num_bloques):
            self.q_completados.get()

        # El Host coge el resultado de la memoria global y lo devuelve
        return self.mem_gpu.res[:tam_datos]
