# Seed: Starting Kit
Below is the standard template to ensure your submission is compatible with our ingestion program.

## Template for ``model.py``
Copy this code into a file named ``model.py``, zip it, and submit it.


```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MyAutoimmuneModel:
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
    return MyAutoimmuneModel()
    
```
## Why use ``predict_proba``?

The scoring program is designed to look for ``predict_proba`` first. 
In pharmacology and toxicity prediction, knowing the confidence of a risk (probability) is often more valuable than a simple *Yes/No* answer. It also provides a more granular ROC-AUC score on the leaderboard.

## Helper to save your results 

You will find in the starting kit if you want to only submit a csv file: 

```python
def save_for_submission(test_probs, private_probs):
    import pandas as pd
    pd.DataFrame(test_probs).to_csv("test_predictions.csv", index=False, header=False)
    pd.DataFrame(private_probs).to_csv("private_test_predictions.csv", index=False, header=False)
    print("Files ready! Zip them and upload.")

```