import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(lam_values, h_values, save_path="results/dual_convergence.png"):
    plt.figure(figsize=(8,5))
    plt.plot(h_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Constraint Violation h(x)")
    plt.title("Dual Gradient Ascent Constraint Violation")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
