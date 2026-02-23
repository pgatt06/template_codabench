from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    This function should return an untrained model instance.
    The ingestion program will call this function to get the model, then train it on the provided training data.
    """
    # Baseline simple : Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model
