from additional_area_method import *
from image_transforms import *
from optics import *
from turbulence import *
from visualization import *
import numpy as np
# Общие параметры
# Число отсчетов
N = 1024
# Размер области, приближаемой двумерным массивом
L = 0.02
# Длина волны пучка
wavelength = 1550e-9 
# Радиус перетяжки
W_0 = 0.002

p_grid = polar_grid(N, L)

# Параметры источника и создание массива, задающего пучок
source = LG_beam(N, L, 0, wavelength, W_0, 0, 0, 0)
source.A = laguerre_gaussian_beam(p_grid, source) / np.sqrt(np.sum(np.abs(
    laguerre_gaussian_beam(p_grid, source))**2))
# Параметры целевого пучка и создание массива, задающего пучок
# Абсолютное значение топологического заряда и радиальный индекс
m, n = 1, 0
target = LG_beam(N, L, 0, wavelength, W_0, n, m, 0)
target.A = laguerre_gaussian_beam(p_grid, target) / np.sqrt(np.sum(np.abs(
    laguerre_gaussian_beam(p_grid, target))**2))

# Параметры ДОЭ и создание экземпляра класса, задающего ДОЭ
f = 0.05
phase_plate = DOE(N, L, f, 0)

# Параметры алгоритма и создание областей
# Допустимая ошибка суммарного расстояния
precision = 1e-6
# Доля мощности целевой моды в полезной области
delta_1 = 0.99
# Параметры турбулентности
# Расстояние между фазовыми экранами
d = 100
# Длина трека
Ltr = 1000
# Параметр Фрида
r_0_1 = 0.01
r_0_2 = 0.02
# Внешний и внутренний масштабы турбулентности
L_0 = 20
l_0 = 0.001
# Создание экземпляра класса, содержащего параметры турбулентности
turbulence_1 = turbulence(N, L, r_0_1, L_0, l_0, d, Ltr)
turbulence_2 = turbulence(N, L, r_0_2, L_0, l_0, d, Ltr)

# Создание трехмерных массивов, содержащих все фазовые экраны для трека
phase_tensor_1 = create_turbulence(int(Ltr/d), turbulence_1)
phase_tensor_2 = create_turbulence(int(Ltr/d), turbulence_2)
# Список для хранения массивов пучков до прохождения через трек
all_results_before = []
# Список для хранения массивов пучков, прошедших через трек с r0 = 1 см
all_results_after_1 = []
# Список для хранения массивов пучков, прошедших через трек с r0 = 2 см
all_results_after_2 = []

# Радиусы основной и дополнительной областей
R1 = find_radius(source.n, source.m, source.sigma_0, delta_1)
R2 = R1*1.86
L1, L2 = create_areas(N, L, R1, R2)

print("Гаусс")
print(f"R1 = {R1}, R2 = {R2}")
result_after_turbulence_1 = propagation(source.A, phase_plate, turbulence_1, 
                                        wavelength, phase_tensor_1)
result_after_turbulence_2 = propagation(source.A, phase_plate, turbulence_2, 
                                        wavelength, phase_tensor_2)
all_results_before.append(source.A)
all_results_after_1.append(mask(result_after_turbulence_1, L1))
all_results_after_2.append(mask(result_after_turbulence_2, L1))
print(correlation(result_after_turbulence_1, source.A, L1))
print(correlation(result_after_turbulence_2, source.A, L1))
np.save('turbulence.npy', phase_tensor_1)
# Создаем массивы для всех мод
modes = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

for i, (m, n) in enumerate(modes):
    if m == 0 and n == 0:
        # Гауссов пучок уже обработан
        continue
    
    R1 = find_radius(n, m, W_0, delta_1)
    R2 = R1*1.86
    L1, L2 = create_areas(N, L, R1, R2) 

    print(f"ГЛ{m:02d}")
    target = LG_beam(N, L, 0, wavelength, W_0, n, m, 0)
    target.A = laguerre_gaussian_beam(p_grid, target) / np.sqrt(np.sum(np.abs(
        laguerre_gaussian_beam(p_grid, target))**2))
    
    phase_plate.T = AAM(source, target, phase_plate, L1, L2, precision)
    result = DOE_propagation(source, phase_plate, L1)
    print(f"R1 = {R1}")
    print(correlation(result, target.A, L1))
    result_after_turbulence_1 = propagation(result, phase_plate, turbulence_1, wavelength, 
                                            phase_tensor_1)
    result_after_turbulence_2 = propagation(result, phase_plate, turbulence_2, wavelength,
                                            phase_tensor_2)
    print(correlation(result_after_turbulence_1, target.A, L1))
    print(correlation(result_after_turbulence_2, target.A, L1))
    all_results_before.append(result)
    all_results_after_1.append(mask(result_after_turbulence_1, L1))
    all_results_after_2.append(mask(result_after_turbulence_2, L1))

# Теперь преобразуем списки в 3D массивы
results_before_3d = np.stack(all_results_before)
results_after_1_3d = np.stack(all_results_after_1)
results_after_2_3d = np.stack(all_results_after_2)

# И отображаем все сразу
plot_propagation(L, results_before_3d, results_after_1_3d, results_after_2_3d,
                 save_svg=True, filename='all_modes.svg')