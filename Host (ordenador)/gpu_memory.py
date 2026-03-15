from multiprocessing import Array, Value


# Estoy suponiendo que la memoria de la GPU está estructurada con los campos que me interesan.
# Es poco realista, pero aceptable para esta simulación.


class GPUMemory:
    def __init__(self, tam_max: int) -> None:
        # Precondición: se asume que tam_max >= 0.
        self.tam_max = Value('i', tam_max)
        self.dato1 = Array('d', tam_max)
        self.dato2 = Array('d', tam_max)
        self.res = Array('d', tam_max)
        self.tam_datos = Value('i', 0)
        self.kernel = Value('i', 0)

        # Para acumular la suma del producto escalar
        self.res_escalar = Value('d', 0.0)

        # Para trabajar con matrices
        self.filas = Value('i', 0)
        self.columnas = Value('i', 0)

        # Espacio para guardar el radio de difuminado
        self.radio = Value('i', 1)
