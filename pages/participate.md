# How to participate

To participate in the **Drug Induced Autoimmunity Prediction challenge**, you must submit a Python implementation of your predictive model. Your model will be trained and evaluated on molecular descriptors generated via RDKit.

## 1. Submission Format

You must submit a **ZIP file** containing a single file named model.py. This file must define a class named Model.

### Mandatory Class Structure
In your ``model.py``, you must define a function ``get_model()`` that returns an object (class or pipeline) with standard Scikit-Learn methods.
Your Model class must implement the following methods:

```python

def get_model():
    # This function must return your model object
    return Model()

class Model:
    def __init__(self):
        # Initialize your model (e.g., RandomForest, MLP, etc.)
        pass

    def fit(self, X, y):
        # X: DataFrame of RDKit descriptors
        # y: Series of target labels (0 or 1)
        pass

    def predict(self, X_test):
        # IMPORTANT: To optimize the ROC-AUC score, this method 
        # should return PROBABILITIES or continuous scores.
        # Returns: numpy array or list of floats
        pass

````

## 2. Evaluation Metrics

Your submission will be evaluated based on **two main metrics**:

**Accuracy**: Measures the proportion of correct classifications.

**ROC-AUC (Primary Metric)**: Measures the ability of your model to distinguish between classes.

*Note*: Since the scoring program uses ```roc_auc_score```, ensure your ``predict()`` function returns confidence scores (probabilities) rather than hard labels to allow for a precise AUC calculation.

## 3. Technical Constraints

**Language:** Python 

**Libraries:** You may use standard ML libraries (Scikit-learn, PyTorch, TensorFlow, NumPy, Pandas). Ensure your code does not require internet access during execution.

**Time Limit:** The training and inference time are tracked and will be included in your final metadata.

**Robustness:** If your model generates NaN values, the scoring program will fill them with a default value (-10), which will heavily penalize your Accuracy and AUC. Ensure your predict() method handles missing data or edge cases.



## 4. Automated Pipeline
Our ingestion program automates the following steps:

**Data Cleaning**: The SMILES column is automatically removed. You will receive only the numerical RDKit descriptors.

**Scoring Optimization**: If your model has a ````predict_proba()```` or ``decision_function()`` method, it will be used automatically to compute the ROC-AUC. This is highly recommended for better ranking.

**Metadata**: Training and inference times are measured and stored in ````metadata.json````.

## 4. How to submit

* Zip your *model.py* (do not zip the folder, just the file).

* Go to the "Submit / View Results" tab.

* Upload your .zip file.

* Wait for the status to change to *Finished* to see your scores on the leaderboard.


## 5. Documentation & Resources

**Seed Page:** Consult the *Seed page* to download a starting kit.

**Dataset:** Familiarize yourself with the chemical descriptors and biological features provided in the data description.