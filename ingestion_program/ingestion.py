import json
import sys
import time
from pathlib import Path

import pandas as pd

EVAL_SETS = ["test", "private_test"]


def evaluate_model(model, X_test):
    X_test = X_test.drop(columns=["SMILES"], errors="ignore")
    y_pred = model.predict(X_test)
    return pd.DataFrame(y_pred)


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


def main(data_dir, output_dir):
    # Here, you can import info from the submission module, to evaluate the
    # submission
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
        X_test = pd.read_csv(data_dir / eval_set / f"{eval_set}_features.csv")
        res[eval_set] = evaluate_model(model, X_test)
    test_time = time.time() - start
    print("-" * 10)
    duration = train_time + test_time
    print(f"Completed Prediction. Total duration: {duration}")

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w+") as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)
    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.csv"
        res[eval_set].to_csv(filepath, index=False)
    print()
    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(str(Path(__file__).parent.resolve()))

    main(Path(args.data_dir), Path(args.output_dir))