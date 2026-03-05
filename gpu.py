from multiprocessing import Queue
from kernels import *
from sm import SM
from gpu_memory import GPUMemory

if __name__ == '__main__':
    # parámetros de construcción de la GPU y puesta en marcha -----------
    cant_sms = 4
    cant_nucleos_por_sm = 5
    tam_mem_gpu = 1000
    mem_gpu = GPUMemory(tam_mem_gpu)
    tam_mem_sm = 100
    q_bloques = Queue()
    sms = [SM(cant_nucleos_por_sm, mem_gpu, tam_mem_sm, q_bloques)
           for _ in range(cant_sms)]
    for s in sms:
        s.start()

    # tarea: incr ------------------------------------------------------
    # El ordenador dice a la GPU lo que debe ejecutar.
    mem_gpu.kernel.value = INCR
    mem_gpu.tam_datos.value = 501
    mem_gpu.dato1[:501] = [1.2] * 501

    # La GPU divide el trabajo en bloques y los encola.
    for block_start in range(0, mem_gpu.tam_datos.value, cant_nucleos_por_sm):
        block_size = min(cant_nucleos_por_sm, mem_gpu.tam_datos.value - block_start)
        q_bloques.put((block_start, block_size))
    for _ in range(cant_sms):  # un None por SM
        q_bloques.put(None)

    # Espera a que todos los SMs acaben e imprime el resultado.
    for sm in sms:
        sm.join()
    print('incr        :', mem_gpu.res[:10])
    print('rdo esperado: [2.2, 2.2, 2.2, ...]')
