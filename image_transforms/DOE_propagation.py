import additional_area_method as aam
import numpy as np
from .angular_spectrum_method import ASM

def DOE_propagation(source, DOE, L1):
    # Функция для моделирования распространения пучка источника на фокусное расстояние
    # после прохождения ДОЭ
    return aam.mask(ASM(source.A * np.exp(1j * np.angle(DOE.T)), DOE,
                             source.wavelength, DOE.f), L1)
