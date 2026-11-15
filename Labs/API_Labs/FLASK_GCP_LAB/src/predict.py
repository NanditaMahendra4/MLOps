import numpy as np
import joblib

model = joblib.load("model/model.pkl")

FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280_od315", "proline",
]

def predict_wine(values):
    """values: dict with the 13 feature names above."""
    row = np.array([[float(values[f]) for f in FEATURES]])
    pred = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row)[0].max())
    return pred, confidence