import numpy as np

class polar_grid:
    # Класс, который будет хранить полярную сетку, необходимую для вычисления
    # комплексной амплитуды пучка
    def __init__(self, N, L):
        # Создание координатной сетки в декартовых координатах
        x = np.linspace(- L/2, L/2, N, endpoint=False)
        y = np.linspace(- L/2, L/2, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Преобразование в полярные координаты
        self.r = np.sqrt(X**2 + Y**2)
        self.phi = np.arctan2(Y, X)  # от -π до π