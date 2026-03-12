from multiprocessing import Queue
from kernels import *
from sm import SM
from gpu_memory import GPUMemory

# Parámetros de construcción de la GPU y puesta en marcha:
CANT_SMS: int = 4
CANT_NUCLEOS_POR_SM: int = 5
TAM_MEM_GPU: int = 1000
TAM_MEM_SM: int = 100
TAM_DATOS_USADOS: int = 501

if __name__ == '__main__':

    mem_gpu = GPUMemory(TAM_MEM_GPU)
    q_bloques = Queue()
    sms = [SM(CANT_NUCLEOS_POR_SM, mem_gpu, TAM_MEM_SM, q_bloques)
           for _ in range(CANT_SMS)]
    for s in sms:
        s.start()

    # Tareas:
    mem_gpu.kernel.value = DIFUMINAR_MAT # Seleccionar la tarea en el kernel
    mem_gpu.tam_datos.value = TAM_DATOS_USADOS

    if mem_gpu.kernel.value == INCR:
        mem_gpu.dato1[:501] = [1.2] * TAM_DATOS_USADOS

    elif mem_gpu.kernel.value == SUMAR or mem_gpu.kernel.value == ESCALAR:
        mem_gpu.dato1[:501] = [1.0] * TAM_DATOS_USADOS
        mem_gpu.dato2[:501] = [2.5] * TAM_DATOS_USADOS

    elif mem_gpu.kernel.value == DIFUMINAR:
        # Preparamos el vector con el "pico"
        vector_difuminado = [10.0] * TAM_DATOS_USADOS
        vector_difuminado[2] = 60.0  # el pico en la posición 2
        mem_gpu.dato1[:501] = vector_difuminado

    elif mem_gpu.kernel.value == DIFUMINAR_MAT:
        # Datos de entrada
        valores = [10, 10, 10, 10,
                   10, 10, 60, 10,
                   10, 10, 10, 10]
        tam_datos = len(valores)
        for i in range(tam_datos):
            mem_gpu.dato1[i] = valores[i]
        mem_gpu.filas.value = 3
        mem_gpu.columnas.value = 4
    # La GPU divide el trabajo en bloques y los encola.
    for block_start in range(0, mem_gpu.tam_datos.value, CANT_NUCLEOS_POR_SM):
        block_size = min(CANT_NUCLEOS_POR_SM, mem_gpu.tam_datos.value - block_start)
        q_bloques.put((block_start, block_size))
    for _ in range(CANT_SMS):  # un None por SM
        q_bloques.put(None)

    # Espera a que todos los SMs acaben e imprime el resultado.
    for sm in sms:
        sm.join()

    if mem_gpu.kernel.value == INCR:
        print('incr        :', [round(x, 2) for x in mem_gpu.res[:10]])
        print('rdo esperado: [2.2, 2.2, 2.2, 2.2, 2.2, ...]')

    elif mem_gpu.kernel.value == SUMAR:
        print('sumar       :', [round(x, 2) for x in mem_gpu.res[:10]])
        print('rdo esperado: [3.5, 3.5, 3.5, 3.5, 3.5, ...]')

    elif mem_gpu.kernel.value == ESCALAR:
        suma_final = sum(mem_gpu.res[:mem_gpu.tam_datos.value])
        print('escalar (parciales) :', [round(x, 2) for x in mem_gpu.res[:10]])
        print(f'Producto Escalar Total: {suma_final}')
        print('rdo esperado        : 1252.5 (porque 1.0*2.5*501 = 1252.5)')

    elif mem_gpu.kernel.value == DIFUMINAR:
        # Imprimimos los 10 primeros redondeados a 2 decimales para leerlos bien
        print('difuminar   :', [round(x, 2) for x in mem_gpu.res[:10]])
        # Como la posición 2 tiene un 60 y el resto 10:
        # - pos 0: solo tiene 3 vecinos útiles (él mismo, pos1, pos2). (10+10+60)/3 = 26.67
        # - pos 1: tiene 4 vecinos útiles (pos0, pos1, pos2, pos3). (10+10+60+10)/4 = 22.5
        # - pos 2: tiene 5 vecinos. (10+10+60+10+10)/5 = 20.0
        # - pos 5: ya no alcanza el pico de 60. (10+10+10+10+10)/5 = 10.0
        print('rdo esperado: [26.67, 22.5, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0]')

    elif mem_gpu.kernel.value == DIFUMINAR_MAT:
        print('incr        :', [round(x, 2) for x in mem_gpu.res[:12]])
        print('rdo esperado: [10.0, 18.33, 18.33, 22.5, 10.0, 15.56, 15.56, 18.33, 10.0, 18.33, 18.33, 22.5]')
