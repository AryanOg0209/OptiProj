import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess(path="data/train.csv"):
    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year.map({2011: 0, 2012: 1})

    y = df["count"]

    numeric_features = ["temp", "atemp", "humidity", "windspeed"]
    categorical_features = [
        "season", "holiday", "workingday", "weather",
        "hour", "dayofweek", "month", "year"
    ]
    # basically polynomial features will capture non linearity adn linear models assume comparable feature scales.
    # basically without scaling temp dominates humidity numerically . 
    # like all the pre-processing is applied inside the pipelines aft3er splitting ,so there is actually no test-train leakage.

    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # X is the feature matrix and y is thee count we require they are split together 
    # train_test_split is from sklearn.model_selection
    # random state 42 will mean ki it ensures consistent treain-test splits acorss runs . 
    # like model seelction was based only on test-set performance .

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features
