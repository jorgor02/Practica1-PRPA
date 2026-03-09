from gpu_memory import GPUMemory
from sm_memory import SMMemory
from threading import Barrier





INCR = 1
SUMAR = 2
DIFUMINAR = 3
ESCALAR = 4



def incr(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + 1

def sumar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier)-> None:
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + gpu_mem.dato2[idx]


def escalar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] * gpu_mem.dato2[idx]


def difuminar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera : Barrier) -> None:
    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        sm_mem.datos[core_id] = gpu_mem.dato1[idx_global]

    barrera.wait()
    
    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        suma = 0.0
        contador = 0
        for rango in range(-2, 3):
            idx_relativo = core_id + rango

            if 0 <= idx_relativo < sm_mem.tam_bloque:
                suma += sm_mem.datos[idx_relativo]
                contador += 1
            else:
                idx_vecino_global = idx_global + rango
                if 0 <= idx_vecino_global < gpu_mem.tam_datos.value:
                    suma += gpu_mem.dato1[idx_vecino_global]
                    contador += 1

        gpu_mem.res[idx_global] = suma / contador



KERNELS = {
    INCR: incr,
    SUMAR: sumar,
    DIFUMINAR: difuminar,
    ESCALAR: escalar,
}
