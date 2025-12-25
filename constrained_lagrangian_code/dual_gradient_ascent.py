import numpy as np
from problem_definition import h
from lagrangian_relaxation import x_of_lambda

def dual_gradient_ascent(lam0=0.0, step=0.1, max_iter=50):
    lam = lam0
    lam_values = []
    h_values = []
    
    for k in range(max_iter):
        x = x_of_lambda(lam)
        violation = h(x)   # this is g'(λ)
        
        lam = lam + step * violation   # ascent step

        lam_values.append(lam)
        h_values.append(violation)

        if abs(violation) < 1e-6:
            break

    return np.array(lam_values), np.array(h_values)

if __name__ == "__main__":
    lam_values, h_values = dual_gradient_ascent()
    print("Final λ:", lam_values[-1])
    print("Final constraint violation:", h_values[-1])
