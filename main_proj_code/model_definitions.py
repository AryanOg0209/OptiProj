import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
# ignoring the harmless warning given by the sklearn
# generate polynomial powers without introducing the cross terms.because Sklearn ka jo PolynomialFeatures it will always include interactionz.
class PowerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        return self
    # like the transformer desint need to learn anythung from data.only here because scikit learn ko ek fit method chahei 

    def transform(self, X):
        X = np.asarray(X)
        feats = [X]
        for p in range(2, self.degree+1):
            feats.append(X**p)
        return np.concatenate(feats, axis=1)
    # transformer here behaves like a scikit-learn preprocessor step used inside a pipeline.
# all the features stacked column-wis so modesl use them.
# build and return regression models with own preprocessing pipeline.
def build_models(numeric_features, categorical_features):

    cat = OneHotEncoder(handle_unknown="ignore")
    # ensures no error during test prediction
    num_linear = Pipeline([("scaler", StandardScaler())])
    # sensitive to feature scale.

    preprocess_linear = ColumnTransformer([
        ("num", num_linear, numeric_features),
        ("cat", cat, categorical_features)
    ])

    models = {}
    # Linear Model
    models["linear"] = Pipeline([
        ("preprocess", preprocess_linear),
        ("regressor", LinearRegression())
    ])
    # no data lekage since preproceessing happends inside the pipeline 
    # agar mai iske andar .fit() call kardun to scaling and encoding bhi sikh jayega and coefficietn ko bhi yad karlega 
    # pipeline se clean safe and reproducible hojayega kaam preprocessing togther kyuki data leakge na ho and training ytesting ka data consistent ho 

    # Polynomial Models without interaction terms
    for d in [2, 3, 4]:
        num_poly = Pipeline([
            ("scaler", StandardScaler()),
            ("powers", PowerFeatures(degree=d))
        ])

        preprocess_poly = ColumnTransformer([
            ("num", num_poly, numeric_features),
            ("cat", cat, categorical_features)
        ])

        models[f"poly_deg_{d}_no_inter"] = Pipeline([
            ("preprocess", preprocess_poly),
            ("regressor", LinearRegression())
        ])

    # Quadratic Model with interaction terms
    # add the interaction terms like(weather*hour)
    num_quad = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])

    preprocess_quad = ColumnTransformer([
        ("num", num_quad, numeric_features),
        ("cat", cat, categorical_features)
    ])

    models["quadratic_with_interactions"] = Pipeline([
        ("preprocess", preprocess_quad),
        ("regressor", LinearRegression())
    ])

    return models
