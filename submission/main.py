import pandas as pd
from submission import get_model
from utils import export_predictions

def to_numeric_features(df):
    df = df.drop(columns=["SMILES"], errors="ignore").copy()
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


X_train = pd.read_csv('dev_phase/input_data/train/train_features.csv')
y_train = pd.read_csv('dev_phase/input_data/train/train_labels.csv')
X_test_pub = pd.read_csv('dev_phase/input_data/test/test_features.csv')
X_test_priv = pd.read_csv('dev_phase/input_data/private_test/private_test_features.csv')


X_train = to_numeric_features(X_train)
X_test_pub = to_numeric_features(X_test_pub)
X_test_priv = to_numeric_features(X_test_priv)


y_train = y_train.iloc[:, 0]
y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)


print("Entraînement du modèle...")
model = get_model()
model.fit(X_train, y_train)

print("Génération des fichiers de prédiction...")
export_predictions(model, X_test_pub, X_test_priv)