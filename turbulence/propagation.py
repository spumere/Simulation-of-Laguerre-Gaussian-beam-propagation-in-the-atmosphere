import numpy as np
from tqdm import tqdm
from image_transforms import ASM

def one_layer_propagation(w, DOE, turbulence_1, wavelength, phase):
    # Формирование фазового экрана
    S = np.exp(1j * phase)
    step = turbulence_1.d/2
    # Шаг 1
    w = ASM(w, DOE, wavelength, step)
    # Шаг 2
    W = w * S
    # Шаг 3
    w = ASM(W, DOE, wavelength, step)
    return w

def propagation(E, DOE, turbulence_1, wavelength, phase_tensor):
    Nlayers = int(turbulence_1.Ltr/turbulence_1.d)
    for i in tqdm(range(Nlayers), desc="Моделирование слоев"):
        E = one_layer_propagation(E, DOE, turbulence_1, wavelength, phase_tensor[i])
    return E