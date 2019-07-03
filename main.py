import numpy as np
import math
from scipy.optimize import minimize
from scipy.misc import derivative

import onedimsearch
import metrics

x0 = [0,0]

def RosenbrockFunction(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)


def GradientRosenbrockFunction(x):
    """The Rosenbrock function's gradient"""
    xm = x [1: -1]
    xm_m1 = x [: - 2]
    xm_p1 = x [2:]
    der = np.zeros_like (x)
    der [1: -1] = 200 * (xm-xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1-xm)
    der [0] = -400 * x [0] * (x [1] -x [0] ** 2) - 2 * (1-x [0])
    der [-1] = 200 * (x [-1] -x [-2] ** 2)
    return der



print(scipy.misc.derivative(RosenbrockFunction, [0,0]))

def func(x):
#     """The Rosenbrock function"""
#     return np.sum(100.0*(x[1:]**2.0-x[:-1])**2.0 + (x[1:]-1)**2.0, axis=0)
    return np.sum((x[1:]-1)**2 + (x[:-1] - 1)**2)

x = np.array([-1.2, 0])

print("\n")
print("----------REAL MIMINUM--------------------------")
print(minimize(func, x))



#
# # X1 - x(k+1) and X2 - x(k)
# def delt_x(X1, X2):
#     X1 = np.array(X1)
#     X2 = np.array(X2)
#     return np.array(X1-X2)
#
# #точность
# eps1 = 0.1
# eps2 = 0.1
#
# MATRIX_X = []
# MATRIX_F = []
#
# old_x = [0,0] #начальная точка
# METRICS = np.eye(2)
# step = 1 #RGR
# DIRECTION = METRICS * GradientRosenbrockFunction(old_x)
# current_x = old_x + step*DIRECTION
#
# while True:
#         step = 1 #функции из РГР на нахождения ламбда
#         METRICS = metrics.metr(current_x, old_x, METRICS)
#         DIRECTION = METRICS * GradientRosenbrockFunction(current_x)
#         new_x = current_x + step*DIRECTION
#
#         check_1 = math.fabs((RosenbrockFunction(new_x) - RosenbrockFunction(current_x)) / RosenbrockFunction(current_x))
#         check_2 = math.fabs(delt_x(current_x) / current_x)
#         if check_1 < eps1 and check_2 < eps2: break
#
#         MATRIX_X.append(current_x)
#         MATRIX_F.append(RosenbrockFunction(current_x))
#
#         old_x = current_x
#         current_x = new_x
#
#
#
#
#
#
#
