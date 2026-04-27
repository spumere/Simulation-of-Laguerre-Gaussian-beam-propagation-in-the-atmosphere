import numpy as np
from scipy.special import genlaguerre
from math import factorial

class LG_beam:
    # Класс, который будет хранить характеристики пучка Лагерра-Гаусса
    def __init__(self, N, L, z, wavelength, sigma_0, n, m, A):
        # Размер двумерного массива отсчетов поперечного амплитудно-фазового распределения пучка
        self.N = N
        # Размер области, приближаемой массивом точек
        self.L = L
        # Координата пучка по оси z
        self.z = z
        # Длина волны пучка
        self.wavelength = wavelength
        # Радиус перетяжки
        self.sigma_0 = sigma_0
        # Радиальный индекс
        self.n = n
        # Абсолютное значение топологического заряда
        self.m = m
        # Массив комплексных амплитуд пучка
        self.A = A

def laguerre_gaussian_beam(polar_grid, beam):
    # Аргумент многочлена Лагерра
    rho_w = 2 * (polar_grid.r**2) / (beam.sigma_0**2)
    # Многочлен Лагерра L_m^n
    laguerre_polynomial = genlaguerre(beam.n, abs(beam.m))
    L = laguerre_polynomial(rho_w)
    # Нормировочная константа
    A_nm = np.sqrt(2*factorial(beam.n)/(np.pi*factorial(beam.n + abs(beam.m))))
    # Комплексная амплитуда пучка Лагерра-Гаусса (z=0)
    PSI_nm = 1/beam.sigma_0 * A_nm * (np.sqrt(2)*polar_grid.r/beam.sigma_0)**abs(beam.m) * L *np.exp(
        1j*beam.m*polar_grid.phi - polar_grid.r**2/beam.sigma_0**2) 
    # Возвращаем массив комплексных чисел, соответствующих комплексной амплитуде пучка в 
    # точках координатной сетки
    return PSI_nm
