from threading import Thread
from gpu_memory import GPUMemory
from kernels import KERNELS
from sm_memory import SMMemory
from threading import Barrier

class Nucleo(Thread):
    def __init__(self, core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera_ini: Barrier, barrera_fin: Barrier, barrera_kernel: Barrier) -> None:
        super().__init__()
        # Precondición: se asume que core_id >= 0 y es único dentro de su SM.

        self.core_id = core_id
        self.gpu_mem = gpu_mem
        self.sm_mem = sm_mem

        self.barrera_ini = barrera_ini
        self.barrera_fin = barrera_fin
        self.barrera_kernel = barrera_kernel

    def run(self) -> None:
        # Esperamos la primera señal del SM antes de evaluar el bucle por primera vez.
        # Esta señal puede ser "hay un bloque nuevo" o "es hora de terminar"
         # Barrera de inicio que espera a que el SM haya iniciado
            #  todos los threads (núcleos)
        self.barrera_ini.wait()

        while not self.sm_mem.terminar:

            kernel_func = KERNELS[self.gpu_mem.kernel.value]

            kernel_func(self.core_id, self.gpu_mem, self.sm_mem, self.barrera_kernel)

            # Barrera final que espera a que todos los núcleos acaben 
            # para que el SM pueda pedir otro bloque
            self.barrera_fin.wait()

            # Fichamos la entrada para el próximo ciclo: nos quedamos a la espera 
            # de un nuevo bloque o de la orden definitiva de terminar.
            self.barrera_ini.wait()
