import requests
import json

# URL de l'API (à modifier si nécessaire)
BASE_URL = "http://localhost:8000"

# Données de test
test_data = {
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

# Vérifier l'état du service
def check_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Statut du service:", response.status_code)
    print(response.json())
    return response.status_code == 200

# Faire une prédiction
def make_prediction(data):
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(data), headers=headers)
    
    print("Statut de la prédiction:", response.status_code)
    if response.status_code == 200:
        result = response.json()
        print("\nRésultat de la prédiction:")
        print(f"Stade prédit: {result['predicted_stage']}")
        print(f"Description: {result['stage_description']}")
        if result['confidence']:
            print(f"Confiance: {result['confidence']:.2%}")
    else:
        print("Erreur:", response.text)
    
    return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    print("Test de l'API de prédiction IRC")
    print("=" * 50)
    
    if check_health():
        print("\nService en ligne, test de prédiction en cours...\n")
        make_prediction(test_data)
    else:
        print("\nService hors ligne. Veuillez vérifier que le serveur est en cours d'exécution.")
