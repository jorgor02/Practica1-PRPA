from threading import Thread
from gpu_memory import GPUMemory
from kernels import KERNELS
from sm_memory import SMMemory


class Nucleo(Thread):
    def __init__(self, core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrier) -> None:
        super().__init__()
        self.core_id = core_id
        self.gpu_mem = gpu_mem
        self.sm_mem = sm_mem
        self.barrier = barrier

    def run(self) -> None:
        self.barrier.wait()

        kernel_func = KERNELS[self.gpu_mem.kernel.value]

        kernel_func(self.core_id, self.gpu_mem, self.sm_mem, self.barrier)

        self.barrier.wait()

