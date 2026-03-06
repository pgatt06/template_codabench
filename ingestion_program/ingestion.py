import json
import sys
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

EVAL_SETS = ["test", "private_test"]


def evaluate_model(model, X_test):
    X_test = X_test.drop(columns=["SMILES"], errors="ignore")

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        y_score = np.asarray(y_score)
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score[:, 0]
        else:
            # if predict_proba returned something unexpected
            y_score = y_score.reshape(-1)
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X_test)).reshape(-1)
    else:
        y_score = np.asarray(model.predict(X_test)).reshape(-1)

    if y_score.shape[0] != X_test.shape[0]:
        raise ValueError(
            f"Bad prediction length: got {y_score.shape[0]} values, expected {X_test.shape[0]}"
        )

    return pd.DataFrame(y_score)


def get_train_data(data_dir):
    data_dir = Path(data_dir)
    training_dir = data_dir / "train"
    X_train = pd.read_csv(training_dir / "train_features.csv")
    y_train = pd.read_csv(training_dir / "train_labels.csv")

    X_train = X_train.drop(columns=["SMILES"], errors="ignore")

    if "Label" in y_train.columns:
        y_train = y_train["Label"]
    else:
        y_train = y_train.iloc[:, 0]

    return X_train, y_train


def try_handle_csv_submission(submission_dir: Path, output_dir: Path) -> bool:
    """
    If the submission contains result CSV files, copy them to output_dir and return True.
    Otherwise return False.
    """
    expected = {"test_predictions.csv", "private_test_predictions.csv"}
    present = {p.name for p in submission_dir.glob("*.csv")}

    if expected.issubset(present):
        print("CSV submission detected: copying prediction files...")
        for name in expected:
            shutil.copy(submission_dir / name, output_dir / name)

        # Create metadata.json so scoring doesn't fail
        (output_dir / "metadata.json").write_text(json.dumps({"train_time": 0, "test_time": 0}))
        return True

    # If user tried CSV mode but files are incomplete, fail early with clear error
    if any(name.endswith("_predictions.csv") for name in present):
        missing = expected - present
        raise FileNotFoundError(
            f"CSV submission incomplete. Missing: {sorted(missing)}. "
            f"Expected exactly: {sorted(expected)}"
        )

    return False


def main(data_dir, output_dir, submission_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Result submission mode
    if try_handle_csv_submission(submission_dir, output_dir):
        return

    # Code submission mode
    from submission import get_model

    X_train, y_train = get_train_data(data_dir)
    print("Training the model")

    model = get_model()

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    print("-" * 10)
    print("Evaluate the model")

    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        X_test = pd.read_csv(Path(data_dir) / eval_set / f"{eval_set}_features.csv")
        res[eval_set] = evaluate_model(model, X_test)
    test_time = time.time() - start

    print("-" * 10)
    duration = train_time + test_time
    print(f"Completed Prediction. Total duration: {duration}")

    # Write output files
    (output_dir / "metadata.json").write_text(json.dumps({"train_time": train_time, "test_time": test_time}))

    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.csv"
        # IMPORTANT: no header, no index
        res[eval_set].to_csv(filepath, index=False, header=False)

    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion program for codabench")
    parser.add_argument("--data-dir", type=str, default="/app/input_data")
    parser.add_argument("--output-dir", type=str, default="/app/output")
    parser.add_argument("--submission-dir", type=str, default="/app/ingested_program")

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(str(Path(__file__).parent.resolve()))

    main(Path(args.data_dir), Path(args.output_dir), Path(args.submission_dir))