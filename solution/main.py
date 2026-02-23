import pandas as pd
from submission import get_model
from utils import export_predictions

# Data Loading
X_train = pd.read_csv('data/train_features.csv')
y_train = pd.read_csv('data/train_labels.csv')
X_test_pub = pd.read_csv('data/test_features.csv')
X_test_priv = pd.read_csv('data/private_test_features.csv')

# Training the model
print("Entraînement du modèle...")
model = get_model()
model.fit(X_train, y_train)

# Exporting predictions to the expected format csv
print("Génération des fichiers de prédiction...")
export_predictions(model, X_test_pub, X_test_priv)