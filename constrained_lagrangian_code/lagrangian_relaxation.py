import numpy as np
from problem_definition import f, h

# Analytical minimizer from solving ∇L w.r.t x = 0
def x_of_lambda(lam):
    x1 = (4 - lam) / 2
    x2 = 1 - lam / 8
    return np.array([x1, x2])

# Dual function derivative = constraint violation
def dual_derivative(lam):
    x = x_of_lambda(lam)
    return h(x)

# Analytical dual optimum
def solve_dual_closed_form():
    # g'(λ) = -5λ/8 = 0 → λ* = 0
    lam_star = 0.0
    x_star = x_of_lambda(lam_star)
    f_star = f(x_star)
    return lam_star, x_star, f_star

if __name__ == "__main__":
    lam_star, x_star, f_star = solve_dual_closed_form()
    print("λ* =", lam_star)
    print("x* =", x_star)
    print("f(x*) =", f_star)
