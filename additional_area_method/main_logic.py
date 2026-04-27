import numpy as np
from .projectors import P1, P2
from image_transforms import ASM
from tqdm import tqdm

def AAM(source, target, DOE, L1, L2, precision):

    # Шаг 0: задаем начальное приближение
    w = np.zeros_like(target.A)
    w[L1] = target.A[L1]
    L1_setminus_L2 = L2 & (~L1)
    mu = np.max(np.abs(target.A[L2]))
    num_random_points = np.sum(L1_setminus_L2)
    random_amplitude = np.random.uniform(0, mu, num_random_points)
    random_phase = np.random.uniform(-np.pi, np.pi, num_random_points)
    random_field = random_amplitude * np.exp(1j * random_phase)
    w[L1_setminus_L2] = random_field

    errors = []
    alpha1 = 1
    alpha2 = 1

    for i in tqdm(range(100), desc="Итерации алгоритма"):
        # Шаг 1: применяем Т2 в плоскости изображения
        w_n = w.copy()
        P2w = P2(target.A, w, L1, L2)
        T2w = w + alpha2 * (P2w - w)
        d1 = np.sqrt(np.sum(np.abs(P2w - w_n)**2))

        # Шаг 2: обратное преобразование Френеля
        W = ASM(T2w, DOE, source.wavelength, DOE.f, inverse = True)
        # Шаг 3: применяем Т1 в плоскости ДОЭ
        w_n = w.copy()
        P1w = P1(source, W, DOE, L2)
        w = T2w + alpha1 * (P1w  - T2w)
        d2 = np.sqrt(np.sum(np.abs(P1w - w_n)**2))
        # Шаг 4: анализируем ошибку и корреляцию
        error = d1 + d2
        errors.append(error)
        if error < precision:
            break
        if i > 2 and abs(errors[i] - errors[i-1]) < precision:
            break
    # Шаг 5: рассчитываем функцию комплексного пропускания ДОЭ
    T =  np.exp(1j*np.angle(ASM(w, DOE, source.wavelength, DOE.f, 
                                              inverse = True)/source.A))
    return T