# Prédiction du départ des clients.

C’est un projet de machine learning. Le but est de créer un programme capable de prédire si un client risque de quitter une entreprise ou non.

En data science, ce problème s’appelle le Customer Churn Prediction

 ## 1. L’idée du projet

Une entreprise veut savoir quels clients risquent de partir.

Par exemple, un client peut partir parce que :

* il n’est pas satisfait ;
* il paie trop cher ;
* il a fait plusieurs réclamations ;
* il a un contrat mensuel sans engagement ;
* il n’a pas de support technique ;
* il utilise une méthode de paiement moins stable.

Le programme va analyser les informations du client et donner une prédiction :
Oui = le client risque de partir
Non = le client ne semble pas à risque

## 2. Les données utilisées

Le projet utilise un fichier CSV :
data/clients_churn.csv

Exemple de données :

IdentifiantClient
Genre
Age
AncienneteMois
TypeContrat
ServiceInternet
SupportTechnique
MethodePaiement
MontantMensuel
MontantTotal
ScoreSatisfaction
NombreReclamations
ClientParti

La colonne la plus importante est : 
ClientParti

C’est la colonne que le modèle doit apprendre à prédire.

Elle contient :

Oui = le client est parti
Non = le client est resté


## 3. Comment fonctionne le projet

Le projet fonctionne en plusieurs étapes.

Étape 1 : charger les données

Le fichier :
src/load_data.py

sert à ouvrir le fichier CSV avec Pandas.

Il transforme le fichier CSV en tableau de données utilisable par Python.

Étape 2 : nettoyer les données

Le fichier :
src/clean_data.py

sert à nettoyer les données avant de les donner au modèle.

Il fait par exemple :

* suppression des lignes en double ;
* suppression des valeurs vides ;
* conversion des nombres ;
* vérification que l’âge est correct ;
* vérification que les montants sont positifs ;
* vérification que le score de satisfaction est entre 1 et 5.

Cette étape est importante parce qu’un modèle de machine learning doit travailler avec des données propres.


Étape 3 : préparer les données

Le fichier :
src/train_model.py

sépare les données en deux parties.

D’un côté, on a les informations utilisées pour prédire :

Genre
Age
AncienneteMois
TypeContrat
ServiceInternet
SupportTechnique
MontantMensuel
ScoreSatisfaction
NombreReclamations

De l’autre côté, on a la réponse à prédire :
ClientParti

Étape 4 : entraîner le modèle

Le modèle utilisé est :
Random Forest Classifier -> C’est un algorithme de classification, Il apprend à partir des anciens clients.
Il regarde les clients qui sont partis et ceux qui sont restés, puis il essaie de trouver des règles.

Par exemple, il peut apprendre que les clients avec :
contrat mensuel + faible satisfaction + plusieurs réclamations

Après l’entraînement, le modèle est sauvegardé dans : 
models/churn_model.pkl

Étape 5 : évaluer le modèle
Après l’entraînement, le programme affiche des résultats dans le terminal

<img width="436" height="221" alt="image" src="https://github.com/user-attachments/assets/74a5f895-9606-4bee-bf09-aad9918d98a6" />

Accuracy : 0.64 --> Cela veut dire que le modèle prédit correctement environ 64 % des cas sur les données de test.

une matrice de confusion : [[57 39]
 [33 71]]
 --> 57 = clients restés bien prédits comme restés
39 = clients restés prédits par erreur comme partis
33 = clients partis prédits par erreur comme restés
71 = clients partis bien prédits comme partis


Étape 6 : faire une prédiction

Le fichier : src/predict.py

sert à charger le modèle sauvegardé et à faire une prédiction sur un nouveau client.

Il reçoit les informations d’un client, par exemple :
 Age : 35
AncienneteMois : 12
TypeContrat : Mensuel
ScoreSatisfaction : 2
NombreReclamations : 3

Puis il retourne : Oui ou Non

Étape 7 : interface avec Streamlit

Le fichier : app/dashboard.py

crée une interface web simple avec Streamlit.



