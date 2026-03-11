from gpu_memory import GPUMemory
from sm_memory import SMMemory
from threading import Barrier


INCR = 1
SUMAR = 2
DIFUMINAR = 3
ESCALAR = 4



def incr(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
       # Precondición: se asume core_id >= 0, y que las memorias están inicializadas.
       if core_id < sm_mem.tam_bloque: 
       # Esta comprobación es por si el ultimo bloque  tiene un número menor
       # de elementos que los anteriores, para evitar que los núcleos sobrantes
       # accedan a memoria fuera de rango

        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + 1

def sumar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier)-> None:
    # Precondición: se asume que core_id >= 0, y que las memorias están inicializadas.
    # gpu_mem.dato1 y dato2 deben tener el mismo tamaño.
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + gpu_mem.dato2[idx]


def escalar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    # Precondición: se asume que core_id >= 0, y que las memorias están inicializadas.
    # gpu_mem.dato1 y dato2 deben tener el mismo tamaño.
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] * gpu_mem.dato2[idx]

""" PREGUNTAR POR ESCALAR SI LA SUMA SE HACE EN LA GPU O EN PARALELO
"""

def difuminar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera : Barrier) -> None:
    # Precondición: se asume que core_id >= 0, y que las memorias están inicializadas.
    # gpu_mem.dato1 y dato2 deben tener el mismo tamaño.
    # Además la barrera debe estar inicializada con el total de núcleos del SM.
    
    #Copiamos el bloque de datos de la memoria global a la memoria compartida del SM
    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        sm_mem.datos[core_id] = gpu_mem.dato1[idx_global]

    #Esperamos a que todos los núcleos hayan copiado su dato a la memoria compartida
    # para coninuar con los cálculos del difuminado, ya que cada núcleo necesita 
    # acceder a los datos de sus vecinos
    barrera.wait()
    
    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        suma = 0.0
        contador = 0
        for rango in range(-2, 3):
            idx_relativo = core_id + rango

            # Si el vecino está en el mismo bloque, leemos rápidamente de la memoria local
            if 0 <= idx_relativo < sm_mem.tam_bloque:
                suma += sm_mem.datos[idx_relativo]
                contador += 1
            # Si el vecino está fuera del bloque, se lee desde la memoria global del gpu
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
