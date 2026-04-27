import numpy as np
from image_transforms import ASM

def P1(source, current, DOE, L2):
    result = ASM(np.abs(source.A)*np.exp(1j*np.angle(current)), DOE, 
                               source.wavelength, DOE.f)
    result[~L2] = 0
    return result

def P2(target, current, L1, L2):
    result = np.zeros_like(current)
    # В области L: берем амплитуду и фазу целевого поля
    result[L1] = target[L1]
    # В области L'/L: берем текущее поле
    L1_setminus_L2 = L2 & (~L1)
    result[L1_setminus_L2] = current[L1_setminus_L2]
    # Вне L': уже 0
    return result