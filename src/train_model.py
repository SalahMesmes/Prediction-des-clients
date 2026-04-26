from pathlib import Path
import sys

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.load_data import load_data
from src.clean_data import clean_data


DATA_PATH = ROOT_DIR / "data" / "clients_churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"


def train_model():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=["IdentifiantClient", "ClientParti"])
    y = df["ClientParti"].map({"Non": 0, "Oui": 1})

    categorical_columns = [
        "Genre",
        "TypeContrat",
        "ServiceInternet",
        "SecuriteEnLigne",
        "SupportTechnique",
        "MethodePaiement",
        "FacturationDematerialisee"
    ]

    numeric_columns = [
        "Age",
        "AncienneteMois",
        "MontantMensuel",
        "MontantTotal",
        "ScoreSatisfaction",
        "NombreReclamations"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=8
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Modèle entraîné avec succès.")
    print(f"Accuracy : {accuracy:.2f}")
    print()
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print()
    print(f"Modèle sauvegardé ici : {MODEL_PATH}")


if __name__ == "__main__":
    train_model()