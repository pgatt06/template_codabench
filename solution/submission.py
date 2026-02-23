from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    Cette fonction est le point d'entrée pour l'ingestion program.
    Elle doit retourner un objet compatible avec l'API sklearn (fit/predict).
    """
    # Baseline simple : Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model
