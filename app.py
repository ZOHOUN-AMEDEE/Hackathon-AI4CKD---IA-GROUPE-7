from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import uvicorn


app = FastAPI(
    title="API de prédiction du stade de l'IRC",
    description="API pour prédire le stade de l'insuffisance rénale chronique à partir de données cliniques",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = joblib.load("best_smote_gradient_boosting.pkl")
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None


class PredictionInput(BaseModel):
    uree: float = Field(..., description="Urée (g/L)", example=0.45)
    creatinine: float = Field(..., description="Créatinine (mg/L)", example=12.5)
    hemoglobine: float = Field(..., description="Hb (g/dL)", example=11.2)
    sodium: float = Field(..., description="Na+ (meq/L)", example=140)
    potassium: float = Field(..., description="K+ (meq/L)", example=4.5)
    calcium: float = Field(..., description="Ca2+ (meq/L)", example=2.3)
    age: int = Field(..., description="Age", example=65)
    sexe: int = Field(..., description="Sexe (0: Femme, 1: Homme)", ge=0, le=1, example=1)
    asthenie: int = Field(..., description="Symptômes/Asthénie (0: Non, 1: Oui)", ge=0, le=1, example=0)
    systole: float = Field(..., description="TA (mmHg)/Systole", example=140)
    etat_general: int = Field(..., description="Etat Général (EG) à l'Admission (sur une échelle)", example=2)


class PredictionOutput(BaseModel):
    predicted_stage: int
    stage_description: str
    confidence: Optional[float] = None


def map_stage(stage_index):
    stages = {
        0: "Stade 1 - Légère",
        1: "Stade 2 - Légère à modérée",
        2: "Stade 3 - Modérée à sévère",
        3: "Stade 4 - Sévère",
        4: "Stade 5 - Terminale (IRCT)"
    }
    return stages.get(stage_index, "Stade inconnu")


@app.get("/")
def read_root():
    return {"message": "API de prédiction du stade de l'IRC. Accédez à /docs pour tester l'API."}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'a pas pu être chargé")
    
    
    features = np.array([
        input_data.uree,
        input_data.creatinine,
        input_data.hemoglobine,
        input_data.sodium,
        input_data.potassium,
        input_data.calcium,
        input_data.age,
        input_data.sexe,
        input_data.asthenie,
        input_data.systole,
        input_data.etat_general
    ]).reshape(1, -1)
    
    try:
        # Faire la prédiction
        prediction = model.predict(features)[0]
        
        # Calculer la confiance si possible
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])
        
        return {
            "predicted_stage": int(prediction),
            "stage_description": map_stage(int(prediction)),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
