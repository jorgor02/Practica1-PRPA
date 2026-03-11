from threading import Thread
from gpu_memory import GPUMemory
from kernels import KERNELS
from sm_memory import SMMemory


class Nucleo(Thread):
    def __init__(self, core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrier) -> None:
        super().__init__()
        # Precondición: se asume que core_id >= 0 y es único dentro de su SM.

        self.core_id = core_id
        self.gpu_mem = gpu_mem
        self.sm_mem = sm_mem
        self.barrier = barrier

    def run(self) -> None:
        # Barrera de inicio que espera a que el SM haya iniciado
        #  todos los threads (núcleos)
        self.barrier.wait()

        kernel_func = KERNELS[self.gpu_mem.kernel.value]

        kernel_func(self.core_id, self.gpu_mem, self.sm_mem, self.barrier)

        # Barrera final que espera a que todos los núcleos acaben 
        # para que el SM pueda pedir otro bloque
        self.barrier.wait()

