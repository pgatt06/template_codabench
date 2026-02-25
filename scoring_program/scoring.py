import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

EVAL_SETS = ["test", "private_test"]


def read_targets(path: Path) -> pd.DataFrame:
    raw_targets = pd.read_csv(path, header=None)
    cleaned_targets = pd.to_numeric(raw_targets.iloc[:, 0], errors="coerce").dropna().astype(int)
    return cleaned_targets.to_frame()


def compute_accuracy(predictions, targets, threshold=0.5):
    y_score = predictions.iloc[:, 0].astype(float).fillna(0.0).to_numpy()
    y_true = targets.iloc[:, 0].astype(int).to_numpy()

    y_pred = (y_score >= threshold).astype(int)
    return (y_pred == y_true).mean()


def compute_roc_auc(predictions, targets):
    y_score = predictions.iloc[:, 0].astype(float).fillna(0.0).to_numpy()
    y_true = targets.iloc[:, 0].astype(int).to_numpy()

    if pd.Series(y_true).nunique() < 2:
        return None

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f"Scoring {eval_set}")

        predictions = pd.read_csv(prediction_dir / f"{eval_set}_predictions.csv", header=None)
        targets = read_targets(reference_dir / f"{eval_set}_labels.csv")

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Prediction length mismatch: got {predictions.shape[0]}, "
                f"expected {targets.shape[0]}"
            )

        if predictions.shape[1] != 1:
            raise ValueError("Predictions must contain exactly one column.")

        # Accuracy (colonnes existantes)
        scores[eval_set] = float(compute_accuracy(predictions, targets))

        # ROC-AUC (nouvelles clés)
        auc = compute_roc_auc(predictions, targets)
        scores[f"{eval_set}_roc_auc"] = auc

    # Add train and test times in the score
    json_durations = (prediction_dir / "metadata.json").read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scoring program for codabench")
    parser.add_argument("--reference-dir", type=str, default="/app/input/ref", help="")
    parser.add_argument("--prediction-dir", type=str, default="/app/input/res", help="")
    parser.add_argument("--output-dir", type=str, default="/app/output", help="")

    args = parser.parse_args()

    main(Path(args.reference_dir), Path(args.prediction_dir), Path(args.output_dir))
