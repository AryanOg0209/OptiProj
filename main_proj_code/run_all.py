from preprocess import load_and_preprocess
from model_definitions import build_models
from train_and_evaluate import train_and_evaluate
from plot_results import generate_plots
import os

os.makedirs("../results", exist_ok=True)

X_train, X_test, y_train, y_test, numf, catf = load_and_preprocess("data/train.csv")
models = build_models(numf, catf)
train_and_evaluate(models, X_train, X_test, y_train, y_test)

print("\nGenerating Actual vs Predicted plots...")
generate_plots()
