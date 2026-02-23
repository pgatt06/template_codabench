import pandas as pd
import os

def export_predictions(model, X_test_public, X_test_private, output_path="./"):
    """
    Génère les fichiers CSV compatibles avec le format attendu par le scoring.
    """
    # Checking if the model has predict_proba or decision_function for better scoring
    if hasattr(model, "predict_proba"):
        pred_public = model.predict_proba(X_test_public)[:, 1]
        pred_private = model.predict_proba(X_test_private)[:, 1]
    else:
        pred_public = model.predict(X_test_public)
        pred_private = model.predict(X_test_private)
    
    # Exporting predictions to the expected format csv
    pd.DataFrame(pred_public).to_csv(os.path.join(output_path, "test_predictions.csv"), index=False, header=False)
    pd.DataFrame(pred_private).to_csv(os.path.join(output_path, "private_test_predictions.csv"), index=False, header=False)
    
    print(f"Fichiers générés dans {output_path}")