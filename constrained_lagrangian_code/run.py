import os
from lagrangian_relaxation import solve_dual_closed_form
from dual_gradient_ascent import dual_gradient_ascent
from plots import plot_convergence

# Making the results directory
os.makedirs("results", exist_ok=True)

# 1. Analytical solution
lam_star, x_star, f_star = solve_dual_closed_form()

with open("results/solution.txt", "w") as f:
    f.write("Analytical Lagrangian Relaxation Solution\n")
    f.write("------------------------------------------\n")
    f.write(f"lam* = {lam_star}\n")
    f.write(f"x* = ({x_star[0]}, {x_star[1]})\n")
    f.write(f"f(x*) = {f_star}\n")

# 2. Numerical dual ascent
lam_vals, h_vals = dual_gradient_ascent(lam0=2.0, step=0.3, max_iter=40)

# Plotting the graph and saving the results
plot_convergence(lam_vals, h_vals, save_path="results/dual_convergence.png")

print("Results generated successfully!")
print("Check the 'results/' folder.")
