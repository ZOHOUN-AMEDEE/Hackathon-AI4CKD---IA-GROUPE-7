# API de prédiction du stade de l'IRC

Cette application FastAPI permet de prédire le stade de l'insuffisance rénale chronique (IRC) à partir de données cliniques, en utilisant un modèle Gradient Boosting entraîné avec SMOTE pour gérer le déséquilibre des classes.

## Prérequis

- Python 3.9+
- Le fichier du modèle entraîné: `best_smote_gradient_boosting.pkl`

## Installation

1. Clonez ce dépôt
2. Installez les dépendances:

```bash
pip install -r requirements.txt
```

## Lancement de l'application

```bash
uvicorn app:app --reload
```

L'API sera accessible à l'adresse http://localhost:8000

## Documentation et test de l'API

La documentation Swagger est disponible à l'adresse http://localhost:8000/docs. Vous pouvez tester l'API directement via cette interface.

## Variables d'entrée

L'API attend les données suivantes:

- `uree`: Urée (g/L)
- `creatinine`: Créatinine (mg/L)
- `hemoglobine`: Hb (g/dL)
- `sodium`: Na+ (meq/L)
- `potassium`: K+ (meq/L)
- `calcium`: Ca2+ (meq/L)
- `age`: Age du patient
- `sexe`: Sexe (0: Femme, 1: Homme)
- `asthenie`: Symptômes/Asthénie (0: Non, 1: Oui)
- `systole`: TA (mmHg)/Systole
- `etat_general`: Etat Général (EG) à l'Admission

## Exemple de requête

```json
{
  "uree": 0.5,
  "creatinine": 15.0,
  "hemoglobine": 10.5,
  "sodium": 138.0,
  "potassium": 4.8,
  "calcium": 2.1,
  "age": 70,
  "sexe": 1,
  "asthenie": 1,
  "systole": 150.0,
  "etat_general": 3
}
```

## Utilisation via Docker

Vous pouvez aussi lancer l'application via Docker:

```bash
# Construire l'image
docker build -t ckd-prediction-api .

# Lancer le conteneur
docker run -p 8000:8000 ckd-prediction-api
```

## Test du service

Pour tester rapidement si l'API fonctionne correctement, exécutez le script de test:

```bash
python test_api.py
```

## Interprétation des résultats

Le modèle prédit un stade de l'IRC parmi les suivants:
- Stade 1: Légère
- Stade 2: Légère à modérée
- Stade 3: Modérée à sévère
- Stade 4: Sévère
- Stade 5: Terminale (IRCT)

La réponse inclut également un score de confiance du modèle pour cette prédiction.
