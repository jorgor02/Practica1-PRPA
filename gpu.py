from multiprocessing import Queue
from kernels import *
from sm import SM
from gpu_memory import GPUMemory


# parámetros de construcción de la GPU y puesta en marcha -----------
CANT_SMS: int = 4
CANT_NUCLEOS_POR_SM: int = 5
TAM_MEM_GPU: int = 1000
TAM_MEM_SM: int = 100
TAM_DATOS_USADOS: int = 501 

if __name__ == '__main__':
    
    mem_gpu = GPUMemory(TAM_MEM_GPU)
    q_bloques = Queue()

    # Arrancamos los SMs (procesos)
    sms = [SM(CANT_NUCLEOS_POR_SM, mem_gpu, TAM_MEM_SM, q_bloques)
           for _ in range(CANT_SMS)]
    for s in sms:
        s.start()

    # El ordenador dice a la GPU la tarea que debe ejecutar.
    mem_gpu.kernel.value = DIFUMINAR
    mem_gpu.tam_datos.value = TAM_DATOS_USADOS

    if mem_gpu.kernel.value == INCR:
        mem_gpu.dato1[TAM_DATOS_USADOS] = [1.2] * TAM_DATOS_USADOS
        
    elif mem_gpu.kernel.value == SUMAR or mem_gpu.kernel.value == ESCALAR:
        mem_gpu.dato1[:TAM_DATOS_USADOS] = [1.0] * TAM_DATOS_USADOS
        mem_gpu.dato2[:TAM_DATOS_USADOS] = [2.5] * TAM_DATOS_USADOS

    elif mem_gpu.kernel.value == DIFUMINAR:
        # Preparamos el vector con un máximo en la posición 2 y el resto con 
        # valores bajos para que se note el efecto del difuminado.
        vector_difuminado = [10.0] * TAM_DATOS_USADOS
        vector_difuminado[2] = 60.0 # el pico en la posición 2
        mem_gpu.dato1[:TAM_DATOS_USADOS] = vector_difuminado

    """
    PREGUNTAR SI PODEMOS PONER NOSOTROS LOS VECTORES QUE QUERRAMOS
    PARA EL EJEMPLO
    """

    # La GPU divide el trabajo en bloques y los encola.
    for block_start in range(0, mem_gpu.tam_datos.value, CANT_NUCLEOS_POR_SM):
        #Aqui usamos min para que en el último bloque no nos pasemos del tamaño de datos a procesar, 
        # que puede no ser del tamaño de CANT_NUCLEOS_POR_SM
        block_size = min(CANT_NUCLEOS_POR_SM, mem_gpu.tam_datos.value - block_start)
        q_bloques.put((block_start, block_size))
    # Señal de finalización (centinela): ponemos un None por cada SM al final
    # para indicarles que terminen su bucle.
    for _ in range(CANT_SMS): 
        q_bloques.put(None)

    # Espera a que todos los SMs acaben e imprime el resultado.
    for sm in sms:
        sm.join()


    """
    PREGUNTAR POR LOS DECIMALES Y SI HACE FALTA REEDONDEAR EN LOS RESULTADOS
    """
    if mem_gpu.kernel.value == INCR:
        print('incr        :', mem_gpu.res[:10])
        print('rdo esperado: [2.2, 2.2, 2.2, 2.2, 2.2, ...]')
        
    elif mem_gpu.kernel.value == SUMAR:
        print('sumar       :', mem_gpu.res[:10])
        print('rdo esperado: [3.5, 3.5, 3.5, 3.5, 3.5, ...]')
        
    elif mem_gpu.kernel.value == ESCALAR:
        suma_final = sum(mem_gpu.res[:mem_gpu.tam_datos.value])
        print('escalar (parciales) :', mem_gpu.res[:10])
        print(f'Producto Escalar Total: {suma_final}')
        print('rdo esperado        : 1252.5 (porque 1.0*2.5*501 = 1252.5)')
        
    elif mem_gpu.kernel.value == DIFUMINAR:
        print('difuminar   :', mem_gpu.res[:10])
        print('rdo esperado: [26.67, 22.5, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0]')