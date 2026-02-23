# Seed: Starting Kit
Below is the standard template to ensure your submission is compatible with our ingestion program.

You can find it in our **Starting Kit** provided.

## Template for ``model.py``
Copy this code into a file named ``model.py``, zip it, and submit it.


```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        # Initialize your classifier
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        """
        X: pandas DataFrame of RDKit descriptors (SMILES are already removed)
        y: target labels
        """
        self.clf.fit(X, y)

    def predict(self, X):
        """
        Used as fallback if predict_proba is not available.
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Recommended: Returns probabilities for the ROC-AUC score.
        """
        return self.clf.predict_proba(X)

def get_model():
    """
    This is the entry point for the competition ingestion program.
    """
    return Model()
    
```
## Why use ``predict_proba``?

The scoring program is designed to look for ``predict_proba`` first. 
In pharmacology and toxicity prediction, knowing the confidence of a risk (probability) is often more valuable than a simple *Yes/No* answer. It also provides a more granular ROC-AUC score on the leaderboard.

## Helper to save your results 

You will find in the starting kit this function if you want to only submit a csv file: 

```python
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
    
    print(f"Files ready in {output_path}")
```