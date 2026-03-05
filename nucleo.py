from threading import Thread
from gpu_memory import GPUMemory
from kernels import KERNELS
from sm_memory import SMMemory


class Nucleo(Thread):
    def __init__(self, core_id, gpu_mem: GPUMemory, sm_mem: SMMemory, barrier) -> None:
        super().__init__()
        self.core_id = core_id
        self.gpu_mem = gpu_mem
        self.sm_mem = sm_mem
        self.barrier = barrier
        self.continuar = True

    def run(self) -> None:
        while self.continuar:
            self.barrier.wait()

            if not self.continuar:
                break
            kernel_func = KERNELS[self.gpu_mem.kernel.value]

            kernel_func(self.core_id, self.gpu_mem, self.sm_mem, self.barrier)

            self.barrier.wait()

