from gpu_memory import GPUMemory
from sm_memory import SMMemory

def incr(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + 1

INCR = 1
SUMAR = 2
DIFUMINAR = 3
ESCALAR = 4

KERNELS = {
    INCR: incr,
    #SUMAR: sumar,
    #DIFUMINAR: difuminar,
    #ESCALAR: escalar,
}
