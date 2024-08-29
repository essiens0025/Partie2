from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Charger le modèle ML sauvegardé
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Définir les noms des espèces pour les prédictions
species_names = ['setosa', 'versicolor', 'virginica']

# Initialiser l'application FastAPI
app = FastAPI()

# Définir le modèle de données pour la requête
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get('/')
def index():
    return {'message': 'Hello, stranger from github'}

@app.get("/predict")
def predict(features: IrisFeatures):
    # Convertir les caractéristiques en format numpy
    data = np.array([
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]).reshape(1, -1)

    # Faire la prédiction
    prediction = model.predict(data)
    predicted_index = prediction[0]
    probabilities = model.predict_proba(data)[0]
    species = species_names[prediction[0]]
    probability = probabilities[predicted_index]

    return {
        "prediction": species,
        "probability": probability
    }
