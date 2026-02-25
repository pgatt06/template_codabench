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

Your submission will be evaluated using two complementary metrics:

### **Accuracy**

Accuracy measures the proportion of correctly classified samples.

Predicted scores are converted into class labels using a threshold of **0.5**:

- score ≥ 0.5 → class 1  
- score < 0.5 → class 0  

Accuracy is then computed by comparing these predicted labels with the true labels.

---

### **ROC-AUC (Primary Metric)**

ROC-AUC measures your model’s ability to rank positive samples higher than negative ones, independently of a specific classification threshold.

This metric is computed directly from the predicted continuous scores using `roc_auc_score`.

Because ROC-AUC evaluates ranking quality, your `predict()` method must return **probabilities or continuous confidence scores**, not hard class labels (0 or 1).

---

### **Important Notes**

- Your `predict()` method must return a single-column array or list of numeric values.
- If your model outputs NaN values, they will be automatically replaced by `0` during scoring. This will significantly penalize both Accuracy and ROC-AUC.
- If the evaluation set contains only one class, ROC-AUC cannot be computed and will be reported as `None`.



## 5. How to submit

* **Prepare your ZIP**: Zip your ``submission.py`` (*Code mode*) OR your two .csv files (*Result mode*).
**Warning:** Zip the files directly, not the folder containing them.

* **Upload**: Go to the "Submit / View Results" tab and upload your .zip.

* **Check Results**: Wait for the status to change to Finished. You can then click on "View Scores" to see your performance on the leaderboard.

## 6. Documentation & Resources

**Seed Page:** Consult the *Seed page* to download a starting kit.

**Dataset:** Familiarize yourself with the chemical descriptors and biological features provided in the data description.