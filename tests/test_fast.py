import unittest
import joblib  # ou import pickle si vous avez utilisé pickle pour enregistrer le modèle

# Chargez le modèle sérialisé
model_path = 'iris_model.pkl'

class TestIrisModel(unittest.TestCase):
    def test_model_initialization(self):
        # Utilisez joblib pour charger le modèle
        new_model = joblib.load(model_path)
        # Assurez-vous que le modèle est correctement chargé (vous pouvez adapter ce test selon votre modèle)
        self.assertIsNotNone(new_model)

if __name__ == '__main__':
    unittest.main()
