# Estoy suponiendo que la memoria de cada SM está estructurada con los campos que me interesan.
# Es poco realista, pero aceptable para esta simulación.

class SMMemory:
    def __init__(self, tam: int) -> None:
        self.datos = [0.0] * tam
        self.ini_bloque = 0
        self.tam_bloque = 0
