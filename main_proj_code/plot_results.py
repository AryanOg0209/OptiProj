# plot_results.py
import os
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess
from model_definitions import build_models

def generate_plots():

    # Load data
    X_train, X_test, y_train, y_test, numf, catf = load_and_preprocess("data/train.csv")

    # Build models
    models = build_models(numf, catf)

    # Create folder
    os.makedirs("../results/plots", exist_ok=True)

    for name, model in models.items():
        print(f"Training & plotting: {name}")

        # Train model
        model.fit(X_train, y_train)

        # Predict on test-set
        preds = model.predict(X_test)

        # Plot Actual vs Predicted
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.xlabel("Actual Count")
        plt.ylabel("Predicted Count")
        plt.title(f"Actual vs Predicted – {name}")

        # Save figure
        save_path = f"../results/plots/{name}_actual_vs_pred.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    print("\nAll plots saved in: results/plots/")
