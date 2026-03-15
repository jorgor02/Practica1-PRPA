from gpu import GPU
from kernels import INCR, SUMAR, ESCALAR, DIFUMINAR, DIFUMINAR_MAT

# Constantes de configuración de la simulación
CANT_SMS: int = 4
CANT_NUCLEOS_POR_SM: int = 5
TAM_MEM_GPU: int = 1000
TAM_MEM_SM: int = 100
TAM_DATOS_USADOS: int = 501 

RADIO_DIFUMINADO_1D: int = 2 # Debe ser > 0
RADIO_DIFUMINADO_MAT: int = 1 # Debe ser > 0


if __name__ == '__main__':
    # Creamos el ordenador (host) con la configuración de la GPU
    ordenador = GPU(CANT_SMS, CANT_NUCLEOS_POR_SM, TAM_MEM_GPU, TAM_MEM_SM)
    

    #Ponemos algunos vectores de ejemplo (se pueden modificar):
    
    # INCR
    vec_base = [1.2] * TAM_DATOS_USADOS
    res_incr = ordenador.ejecutar_trabajo_en_gpu(INCR, vec_base)
    print('incr        :', res_incr[:10])
    print('rdo esperado: [2.2, 2.2, 2.2, 2.2, 2.2, ...]\n')
    
    # SUMAR
    vec_a = [1.0] * TAM_DATOS_USADOS
    vec_b = [2.5] * TAM_DATOS_USADOS
    res_sumar = ordenador.ejecutar_trabajo_en_gpu(SUMAR, vec_a, vec_b)
    print('sumar       :', res_sumar[:10])
    print('rdo esperado: [3.5, 3.5, 3.5, 3.5, 3.5, ...]\n')
    
    # PRODUCTO ESCALAR
    ordenador.ejecutar_trabajo_en_gpu(ESCALAR, vec_a, vec_b)
    res_escalar = ordenador.mem_gpu.res_escalar.value # Obtenemos el resultado de gpu_mem
    print(f'producto escalar   : {res_escalar}')
    print('rdo esperado       : 1252.5 (porque 1.0*2.5*501 = 1252.5)\n')
        
    # DIFUMINAR
    # Preparamos el vector con un máximo en la posición 2 y el resto con 
    # valores bajos para que se note el efecto del difuminado.
    vec_base = [10.0] * TAM_DATOS_USADOS
    vec_base[2] = 60.0 # el pico en la posición 2
    res_difuminar = ordenador.ejecutar_trabajo_en_gpu(DIFUMINAR, vec_base, radio=RADIO_DIFUMINADO_1D)
    print(f'difuminar (r = {RADIO_DIFUMINADO_1D})  :', res_difuminar[:10])
    if RADIO_DIFUMINADO_1D == 2:
        print('rdo esperado       : [26.67, 22.5, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0]\n')
    else:
        print('')

    # DIFUMINAR MATRICES
    valores = [10, 10, 10, 10,
               10, 10, 60, 10,
               10, 10, 10, 10]
    # Es necesario especificar las dimensiones de la matriz, ya que se almacena de forma lineal
    filas = 3
    columnas = 4
    res_difu_mat = ordenador.ejecutar_trabajo_en_gpu(DIFUMINAR_MAT, valores, None, filas, columnas, radio=RADIO_DIFUMINADO_MAT)
    print(f'difuminar matrices (r = {RADIO_DIFUMINADO_MAT})  :', [round(x, 2) for x in res_difu_mat[:12]])
    if RADIO_DIFUMINADO_MAT == 1:
        print('rdo esperado                : [10.0, 18.33, 18.33, 22.5, 10.0, 15.56, 15.56, 18.33, 10.0, 18.33, 18.33, 22.5]')
    if RADIO_DIFUMINADO_MAT == 2:
        print('rdo esperado                : [15.56, 14.17, 14.17, 15.56, 15.56, 14.17, 14.17, 15.56, 15.56, 14.17, 14.17, 15.56]')
    if RADIO_DIFUMINADO_MAT > 2:
        print('rdo esperado                : [14.17, 14.17, 14.17, 14.17, 14.17, 14.17, 14.17, 14.17, 14.17, 14.17, 14.17, 14.17]')
    # Apagamos la GPU (detenemos los procesos de los SMs)
    ordenador.apagar()
