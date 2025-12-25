import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
# r^2=-inf-1
# prediction error and explained variance.

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    # key model name and value pipeline object
    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        # preprocessing fit hojayeng features mat par and reg coef using y trian
        # ensures ki training trainijg data par hi hora hai 

        preds = model.predict(X_test)
        # it actually measures generalisation 
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "model": name,
            "test_MSE": mse,
            "test_R2": r2
        })

    df = pd.DataFrame(results)
    df.to_csv("../results/metrics_table.csv", index=False)

    best = df.sort_values("test_MSE").iloc[0]
    # loweest test mse sortings.
    with open("../results/best_model.txt", "w") as f:
        f.write(f"Best model: {best['model']}\n")
        f.write(f"MSE: {best['test_MSE']}\n")
        f.write(f"R2: {best['test_R2']}\n")

    print("\nDONE! Results saved in results/")
    return df

