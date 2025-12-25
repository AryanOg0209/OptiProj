import numpy as np

# Objective function / Cost Function
def f(x):
    x1, x2 = x
    return x1**2 + 4*x2**2 - 4*x1 - 8*x2 + 20

# Gradient of f
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2*x1 - 4
    df_dx2 = 8*x2 - 8
    return np.array([df_dx1, df_dx2])

# Constraint h(x) = 0
def h(x):
    x1, x2 = x
    return x1 + x2 - 3

# Lagrangian L(x, λ)
def L(x, lam):
    return f(x) + lam * h(x)
