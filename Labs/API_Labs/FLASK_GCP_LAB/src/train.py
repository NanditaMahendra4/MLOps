import os
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_training():
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVMs need scaled features, so scaler + SVC go in one pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
    ])
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc:.3f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("Saved trained model to model/model.pkl")

if __name__ == "__main__":
    run_training()