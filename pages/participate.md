# How to participate

To participate in the **Drug Induced Autoimmunity Prediction challenge**, you must submit a Python implementation of your predictive model. Your model will be trained and evaluated on molecular descriptors generated via RDKit.

## 1. Submission Format

You can participate in two ways:

**Code Submission (Recommended):** Submit a **ZIP file** containing ``submission.py``. Our system will train and test your model. 

**Result Submission:** If you prefer to work locally, you can submit a **ZIP file** containing:

* test_predictions.csv

* private_test_predictions.csv

*Note : A function export_predictions() is provided in the ``utils.py`` of the starting kit to help you generate these files correctly (single column, no header, no index).*

## 2. Mandatory Class Structure
In your ``submission.py``, you must define a function ``get_model()`` that returns your model object.
To maximize your *ROC-AUC score*, your class should ideally implement ``predict_proba``.


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

## 3. Evaluation Metrics

Your submission will be evaluated based on **two main metrics**:

**Accuracy**: Measures the proportion of correct classifications.

**ROC-AUC (Primary Metric)**: Measures the ability of your model to distinguish between classes.

*Note*: Since the scoring program uses ```roc_auc_score```, ensure your ``predict()`` function returns confidence scores (probabilities) rather than hard labels to allow for a precise AUC calculation.

## 4. Technical Constraints

**Language:** Python 

**Automated Pipeline**: The SMILES column is automatically removed before the data reaches your fit and predict methods. You will work with numerical RDKit descriptors only.

**Time Limit:** The training and inference time are tracked and will be included in your final metadata.

**Robustness:** If your model generates NaN values, the scoring program will fill them with a default value (-10), which will heavily penalize your Accuracy and AUC. Ensure your ``predict()`` method handles missing data or edge cases.



## 5. How to submit

* **Prepare your ZIP**: Zip your ``submission.py`` (*Code mode*) OR your two .csv files (*Result mode*).
**Warning:** Zip the files directly, not the folder containing them.

* **Upload**: Go to the "Submit / View Results" tab and upload your .zip.

* **Check Results**: Wait for the status to change to Finished. You can then click on "View Scores" to see your performance on the leaderboard.

## 6. Documentation & Resources

**Seed Page:** Consult the *Seed page* to download a starting kit.

**Dataset:** Familiarize yourself with the chemical descriptors and biological features provided in the data description.