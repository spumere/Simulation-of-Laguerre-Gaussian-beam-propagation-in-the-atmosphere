import numpy as np
import aotools

def create_turbulence(N, turbulence):
    phase_tensor = np.empty((N, turbulence.N, turbulence.N), dtype=np.float32)
    for i in range(N):
        phase = aotools.turbulence.phasescreen.ft_phase_screen(turbulence.r_0, turbulence.N, 
                                                           turbulence.dx, turbulence.L_0, 
                                                           turbulence.l_0)
        phase_tensor[i] = phase
    return phase_tensor
