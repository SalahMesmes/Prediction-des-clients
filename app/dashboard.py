import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_churn


st.set_page_config(
    page_title="Prédiction du départ client",
    page_icon="📊",
    layout="centered"
)

st.title("Prédiction du départ client")

st.write(
    "Cette application utilise un modèle de machine learning pour prédire "
    "si un client risque de quitter l'entreprise."
)

st.subheader("Informations du client")

genre = st.selectbox("Genre", ["Homme", "Femme"])

age = st.slider("Âge", 18, 80, 35)

anciennete = st.slider("Ancienneté en mois", 1, 72, 12)

type_contrat = st.selectbox(
    "Type de contrat",
    ["Mensuel", "Un an", "Deux ans"]
)

service_internet = st.selectbox(
    "Service Internet",
    ["Fibre optique", "ADSL", "Aucun"]
)

securite_en_ligne = st.selectbox(
    "Sécurité en ligne",
    ["Oui", "Non"]
)

support_technique = st.selectbox(
    "Support technique",
    ["Oui", "Non"]
)

methode_paiement = st.selectbox(
    "Méthode de paiement",
    ["Chèque électronique", "Carte bancaire", "Virement bancaire", "Chèque postal"]
)

facturation_dematerialisee = st.selectbox(
    "Facturation dématérialisée",
    ["Oui", "Non"]
)

montant_mensuel = st.number_input(
    "Montant mensuel",
    min_value=1.0,
    max_value=200.0,
    value=65.0
)

montant_total = st.number_input(
    "Montant total",
    min_value=1.0,
    max_value=10000.0,
    value=780.0
)

score_satisfaction = st.slider("Score de satisfaction", 1, 5, 3)

nombre_reclamations = st.slider("Nombre de réclamations", 0, 5, 1)

client_data = {
    "Genre": genre,
    "Age": age,
    "AncienneteMois": anciennete,
    "TypeContrat": type_contrat,
    "ServiceInternet": service_internet,
    "SecuriteEnLigne": securite_en_ligne,
    "SupportTechnique": support_technique,
    "MethodePaiement": methode_paiement,
    "FacturationDematerialisee": facturation_dematerialisee,
    "MontantMensuel": montant_mensuel,
    "MontantTotal": montant_total,
    "ScoreSatisfaction": score_satisfaction,
    "NombreReclamations": nombre_reclamations
}

if st.button("Prédire"):
    prediction, probability = predict_churn(client_data)

    st.subheader("Résultat de la prédiction")

    if prediction == "Oui":
        st.error(f"Le client risque de partir. Probabilité : {probability} %")
    else:
        st.success(f"Le client ne semble pas à risque. Probabilité de départ : {probability} %")

    st.write("Données utilisées pour la prédiction :")
    st.dataframe(pd.DataFrame([client_data]), use_container_width=True)