import pandas as pd
from submission import Model

def export_predictions(model, X_test_public, X_test_private, output_path="./"):
    """
    Génère les fichiers CSV compatibles avec le format attendu par le scoring.
    """
    # Prédiction (probabilités recommandées)
    pred_public = model.predict_proba(X_test_public)[:, 1]
    pred_private = model.predict_proba(X_test_private)[:, 1]
    
    # Export
    pd.DataFrame(pred_public).to_csv(f"{output_path}/test_predictions.csv", index=False)
    pd.DataFrame(pred_private).to_csv(f"{output_path}/private_test_predictions.csv", index=False)
    print("Fichiers prêts pour la soumission !")