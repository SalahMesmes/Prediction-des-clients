import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    required_columns = [
        "IdentifiantClient",
        "Genre",
        "Age",
        "AncienneteMois",
        "TypeContrat",
        "ServiceInternet",
        "SecuriteEnLigne",
        "SupportTechnique",
        "MethodePaiement",
        "FacturationDematerialisee",
        "MontantMensuel",
        "MontantTotal",
        "ScoreSatisfaction",
        "NombreReclamations",
        "ClientParti"
    ]

    df = df.dropna(subset=required_columns)

    numeric_columns = [
        "Age",
        "AncienneteMois",
        "MontantMensuel",
        "MontantTotal",
        "ScoreSatisfaction",
        "NombreReclamations"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns)

    df = df[df["Age"] >= 18]
    df = df[df["AncienneteMois"] > 0]
    df = df[df["MontantMensuel"] > 0]
    df = df[df["MontantTotal"] > 0]
    df = df[df["ScoreSatisfaction"].between(1, 5)]
    df = df[df["NombreReclamations"] >= 0]

    return df