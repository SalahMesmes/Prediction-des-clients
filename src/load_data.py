import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable : {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()