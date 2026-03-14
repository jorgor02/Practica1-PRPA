from multiprocessing import Process, Queue
from threading import Barrier
from gpu_memory import GPUMemory
from sm_memory import SMMemory
from nucleo import Nucleo


CENTINELA = None


class SM(Process):
    def __init__(self, cant_nucleos: int, gpu_mem: GPUMemory, tam_mem_sm: int, q_bloques: Queue, q_completados: Queue) -> None:
        super().__init__()
        # Precondición: se asume que cant_nucleos > 0 y tam_mem_sm > 0.
        self.cant_nucleos = cant_nucleos
        self.gpu_mem = gpu_mem
        self.tam_mem_sm = tam_mem_sm
        self.q_bloques = q_bloques
        self.q_completados = q_completados

    def run(self) -> None:

        sm_mem = SMMemory(self.tam_mem_sm)
        sm_mem.terminar = False
        
        barrera_ini = Barrier(self.cant_nucleos + 1) 
        barrera_fin = Barrier(self.cant_nucleos + 1)   
        # Barrera interna exclusiva para las funciones del kernel, 
        # para sincronizar a los núcleos después de cada paso del kernel.
        barrera_kernel = Barrier(self.cant_nucleos)

        nucleos = [
                Nucleo(i, self.gpu_mem, sm_mem, barrera_ini, barrera_fin, barrera_kernel)
                for i in range(self.cant_nucleos)
            ]

        for n in nucleos:
            n.start()


        bloque = self.q_bloques.get()
        # Consumimos bloques de la cola hasta recibir la 
        # señal de que se han acabado (None)
        while bloque is not CENTINELA:

            block_start, block_size = bloque

            sm_mem.ini_bloque = block_start
            sm_mem.tam_bloque = block_size

            # Despertamos a los núcleos
            barrera_ini.wait()
            
            # Esperamos a que todos terminen
            barrera_fin.wait()
            
            
            # Avisamos al Host de que este bloque ha concluido
            self.q_completados.put(True)
    
            bloque = self.q_bloques.get()

        # Si llega el CENTINELA, marcamos la salida y despertamos a los hilos
        sm_mem.terminar = True
        barrera_ini.wait()

        for n in nucleos:
                        n.join()