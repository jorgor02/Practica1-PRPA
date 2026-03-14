from gpu_memory import GPUMemory
from sm_memory import SMMemory
from threading import Barrier

INCR = 1
SUMAR = 2
DIFUMINAR = 3
ESCALAR = 4
DIFUMINAR_MAT = 5


def incr(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    # Precondición: se asume core_id >= 0 y que las memorias están inicializadas.
    if core_id < sm_mem.tam_bloque:
        # Esta comprobación es por si el último bloque tiene un número menor
        # de elementos que los anteriores, para evitar que los núcleos sobrantes
        # accedan a memoria fuera de rango
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + 1


def sumar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    # Precondición: se asume que core_id >= 0 y que las memorias están inicializadas.
    # gpu_mem.dato1 y dato2 deben tener el mismo tamaño.
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + gpu_mem.dato2[idx]


def escalar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    # Precondición: se asume que core_id >= 0 y que las memorias están inicializadas.
    # gpu_mem.dato1 y dato2 deben tener el mismo tamaño.
    if core_id < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + core_id
        gpu_mem.res[idx] = gpu_mem.dato1[idx] * gpu_mem.dato2[idx]


def difuminar(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera: Barrier) -> None:
    # Precondición: se asume que core_id >= 0 y que las memorias están inicializadas.
    # gpu_mem.dato1 y dato2 deben tener el mismo tamaño.
    # Además la barrera debe estar inicializada con el total de núcleos del SM.

    # Copiamos el bloque de datos de la memoria global a la memoria compartida del SM
    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        sm_mem.datos[core_id] = gpu_mem.dato1[idx_global]

    # Esperamos a que todos los núcleos hayan copiado su dato a la memoria compartida
    # para continuar con los cálculos del difuminado, ya que cada núcleo necesita
    # acceder a los datos de sus vecinos
    barrera.wait()

    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        suma = 0.0
        contador = 0
        radio = gpu_mem.radio.value

        for rango in range(-radio, radio + 1):
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


def difuminar_mat(core_id: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera: Barrier) -> None:
    # Precondición: se asume que core_id >= 0 y que las memorias están inicializadas.
    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        sm_mem.datos[core_id] = gpu_mem.dato1[idx_global]

    barrera.wait()

    if core_id < sm_mem.tam_bloque:
        idx_global = sm_mem.ini_bloque + core_id
        suma = 0.0
        contador = 0
        radio = gpu_mem.radio.value

        # Recorremos vecinos 3x3:
        for i in range(-radio, radio + 1):
            for j in range(-radio, radio + 1):
                # Calculamos en que fila y columna estamos:
                fila = idx_global // gpu_mem.columnas.value + i
                col = idx_global % gpu_mem.columnas.value + j

                # Calculamos el índice de cada vecino
                if 0 <= fila < gpu_mem.filas.value and 0 <= col < gpu_mem.columnas.value:
                    vecino_idx = fila * gpu_mem.columnas.value + col

                    # Comprobamos si el vecino pertenece a la matriz y contamos cuantos hay (importante en los bordes)
                    if 0 <= vecino_idx < gpu_mem.tam_datos.value:
                        idx_relativo = vecino_idx - sm_mem.ini_bloque
                        # Si el vecino está dentro de nuestro bloque cargado, leemos de la memoria de SM
                        if 0 <= idx_relativo < sm_mem.tam_bloque:
                            suma += sm_mem.datos[idx_relativo]
                        # Si el vecino pertenece a otro bloque, toca ir a la memoria global (lento)
                        else:
                            suma += gpu_mem.dato1[vecino_idx]  # pyright: ignore[reportOperatorIssue]
                        contador += 1

        if contador > 0:
            gpu_mem.res[idx_global] = suma / contador


KERNELS = {
    INCR: incr,
    SUMAR: sumar,
    DIFUMINAR: difuminar,
    ESCALAR: escalar,
    DIFUMINAR_MAT: difuminar_mat
}
