import numpy as np

def create_areas(N, L_size, R1, R2):
    # Функция для создания полезной L1 и дополнительной L\L1 = L2
    # областей в фокальной плоскости
    x = np.linspace(-L_size/2, L_size/2, N, endpoint=False)
    y = np.linspace(-L_size/2, L_size/2, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    L1 = r <= R1
    L2 = r <= R2
    return L1, L2

def mask(field, L1):
    # Создание маски для визуализации
    masked = np.zeros_like(field)
    masked[L1] = field[L1]
    return masked
