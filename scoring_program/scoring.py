import json
from pathlib import Path

import pandas as pd

EVAL_SETS = ["test", "private_test"]


def compute_accuracy(predictions, targets):
    # prend la 1ère colonne des deux fichiers
    pred = predictions.iloc[:, 0].fillna(-10).to_numpy()
    y = targets.iloc[:, 0].to_numpy()
    return (pred == y).mean()


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')

        predictions = pd.read_csv(
            prediction_dir / f'{eval_set}_predictions.csv'
        )
        targets = pd.read_csv(
            reference_dir / f'{eval_set}_labels.csv'
        )

        scores[eval_set] = float(compute_accuracy(predictions, targets))

    # Add train and test times in the score
    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
