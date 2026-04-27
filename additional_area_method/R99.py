import numpy as np
import scipy as sp
from scipy.special import genlaguerre
from math import factorial

def power_fraction(T, n, m, sigma0=1.0):
    # Возвращает долю мощности F(T) для LG_n^m пучка.
    # T = 2*R1^2 / sigma0^2

    # полином Лагерра
    L = genlaguerre(n, abs(m))
    
    def integrand(t):
        return (t**abs(m)) * (L(t)**2) * np.exp(-t)
    
    # Интеграл от 0 до T
    integral, _ = sp.integrate.quad(integrand, 0, T, limit=200)
    
    # Полная мощность (интеграл от 0 до ∞)
    total = factorial(n + abs(m)) / factorial(n)
    
    return integral / total

def find_radius(n, m, sigma0, delta=0.99, tol=1e-10):
    ## Находит радиус полезной области R1 для пучка Гаусса-Лагерра
    # Поиск верхней границы
    T_high = 10 + 2*abs(m) + 4*n
    while power_fraction(T_high, n, m) < delta:
        T_high *= 2
    
    # Решаем F(T) - delta = 0 методом половинного деления
    T_solution = sp.optimize.bisect(lambda T: power_fraction(T, n, m) - delta, 
                        0, T_high, xtol=tol)
    
    # Переход к физическому радиусу
    R1 = sigma0 * np.sqrt(T_solution / 2.0)
    return R1