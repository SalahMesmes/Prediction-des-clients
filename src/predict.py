from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"


def load_model():
    """
    Charge le modèle entraîné.
    """
    return joblib.load(MODEL_PATH)


def predict_churn(client_data: dict):
    """
    Prédit si un client risque de partir.
    """
    model = load_model()

    df_client = pd.DataFrame([client_data])

    prediction = model.predict(df_client)[0]
    probability = model.predict_proba(df_client)[0][1]

    result = "Oui" if prediction == 1 else "Non"

    return result, round(probability * 100, 2)