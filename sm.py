from multiprocessing import Process, Queue
from threading import Barrier
from gpu_memory import GPUMemory
from sm_memory import SMMemory
from nucleo import Nucleo

class SM(Process):
    def __init__(self, cant_nucleos: int, gpu_mem: GPUMemory, tam_mem_sm: int, q_bloques: Queue) -> None:
        super().__init__()
        self.cant_nucleos = cant_nucleos
        self.gpu_mem = gpu_mem
        self.tam_mem_sm = tam_mem_sm
        self.q_bloques = q_bloques

    def run(self) -> None:

        sm_mem = SMMemory(self.tam_mem_sm)
        bloque = self.q_bloques.get()

        while bloque is not None:

            block_start, block_size = bloque

            sm_mem.ini_bloque = block_start
            sm_mem.tam_bloque = block_size

            barrier = Barrier(self.cant_nucleos)

            nucleos = [
                Nucleo(i, self.gpu_mem, sm_mem, barrier)
                for i in range(self.cant_nucleos)
            ]

            for n in nucleos:
                n.start()

            for n in nucleos:
                n.join()

            bloque = self.q_bloques.get()
