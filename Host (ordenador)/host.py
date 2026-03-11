from gpu import GPU
from kernels import INCR, SUMAR, ESCALAR, DIFUMINAR

# Constantes de configuración de la simulación
CANT_SMS: int = 4
CANT_NUCLEOS_POR_SM: int = 5
TAM_MEM_GPU: int = 1000
TAM_MEM_SM: int = 100
TAM_DATOS_USADOS: int = 501 

if __name__ == '__main__':
    # Creamos el ordenador (host) con la configuración de la GPU
    ordenador = GPU(CANT_SMS, CANT_NUCLEOS_POR_SM, TAM_MEM_GPU, TAM_MEM_SM)
    

    """
    PREGUNTAR POR LOS DECIMALES Y SI HACE FALTA REEDONDEAR EN LOS RESULTADOS
    """
    
    """
    PREGUNTAR SI PODEMOS PONER NOSOTROS LOS VECTORES QUE QUERRAMOS
    PARA EL EJEMPLO
    """
    
    # INCR
    vec_base = [1.2] * TAM_DATOS_USADOS
    res_incr = ordenador.ejecutar_trabajo_en_gpu(INCR, vec_base)
    print('incr        :', res_incr[:10])
    print('rdo esperado: [2.2, 2.2, 2.2, 2.2, 2.2, ...]')
    
    # SUMAR
    vec_a = [1.0] * TAM_DATOS_USADOS
    vec_b = [2.5] * TAM_DATOS_USADOS
    res_sumar = ordenador.ejecutar_trabajo_en_gpu(SUMAR, vec_a, vec_b)
    print('sumar       :', res_sumar[:10])
    print('rdo esperado: [3.5, 3.5, 3.5, 3.5, 3.5, ...]')
    
    # PRODUCTO ESCALAR
    res_escalar_parcial = ordenador.ejecutar_trabajo_en_gpu(ESCALAR, vec_a, vec_b)
    # El host hace la reducción (suma final) de los resultados parciales de la GPU
    suma_final = sum(res_escalar_parcial)
    print(f'producto escalar: {suma_final}')
    print('rdo esperado        : 1252.5 (porque 1.0*2.5*501 = 1252.5)')
        
    # DIFUMINAR
    # Preparamos el vector con un máximo en la posición 2 y el resto con 
    # valores bajos para que se note el efecto del difuminado.
    vec_base = [10.0] * TAM_DATOS_USADOS
    vec_base[2] = 60.0 # el pico en la posición 2
    res_difuminar = ordenador.ejecutar_trabajo_en_gpu(DIFUMINAR, vec_base)
    print('difuminar   :', res_difuminar[:10])
    print('rdo esperado: [26.67, 22.5, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0]')