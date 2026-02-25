import shutil
import uuid
from pathlib import Path

try:
    import docker
except ImportError:
    raise ImportError(
        "The 'docker' package is required to run this script. "
        "Please install it using 'pip install docker'."
    )

REPO = Path(__file__).resolve().parent.parent
IMAGE = "tommoral/template:v1"
EXPECTED_CSV = {"test_predictions.csv", "private_test_predictions.csv"}
TMP_SUBMISSIONS = REPO / ".tmp_submissions"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_submission_variants():
    solution_dir = REPO / "solution"
    code_submission_dir = TMP_SUBMISSIONS / "code_submission"
    csv_submission_dir = TMP_SUBMISSIONS / "csv_submission"

    reset_dir(TMP_SUBMISSIONS)
    reset_dir(code_submission_dir)
    reset_dir(csv_submission_dir)

    copied_csv = set()
    for item in solution_dir.iterdir():
        if item.name.startswith("."):
            continue

        if item.is_dir():
            if item.name == "__pycache__":
                continue
            shutil.copytree(item, code_submission_dir / item.name)
            continue

        if item.name in EXPECTED_CSV:
            shutil.copy2(item, csv_submission_dir / item.name)
            copied_csv.add(item.name)
            continue

        if item.name.endswith("_predictions.csv"):
            # Ignore extra prediction files that are not part of the expected pair.
            continue

        shutil.copy2(item, code_submission_dir / item.name)

    missing_csv = EXPECTED_CSV - copied_csv
    if missing_csv:
        raise FileNotFoundError(
            "CSV submission files are missing in solution/. "
            f"Expected: {sorted(EXPECTED_CSV)}, missing: {sorted(missing_csv)}"
        )

    return code_submission_dir, csv_submission_dir


def run_container(client, *, command: str, name_prefix: str, volumes: list[str]) -> None:
    logs = client.containers.run(
        image=IMAGE,
        command=command,
        remove=True,
        name=f"{name_prefix}-{uuid.uuid4().hex[:8]}",
        user="root",
        volumes=volumes,
    )
    print(logs.decode("utf-8"))


def run_pipeline(client, *, mode: str, submission_dir: Path, ingestion_res: Path, scoring_res: Path) -> None:
    print(f"Running Docker pipeline ({mode})...")
    reset_dir(ingestion_res)
    reset_dir(scoring_res)

    run_container(
        client,
        command="python3 /app/ingestion_program/ingestion.py",
        name_prefix=f"ingestion-{mode}",
        volumes=[
            f"{REPO}/ingestion_program:/app/ingestion_program",
            f"{REPO}/dev_phase/input_data:/app/input_data",
            f"{ingestion_res}:/app/output",
            f"{submission_dir}:/app/ingested_program",
        ],
    )

    run_container(
        client,
        command="python3 /app/scoring_program/scoring.py",
        name_prefix=f"scoring-{mode}",
        volumes=[
            f"{REPO}/scoring_program:/app/scoring_program",
            f"{REPO}/dev_phase/reference_data:/app/input/ref",
            f"{ingestion_res}:/app/input/res",
            f"{scoring_res}:/app/output",
        ],
    )

    scores_path = scoring_res / "scores.json"
    if scores_path.exists():
        print(f"{mode} scores: {scores_path.read_text()}")


if __name__ == "__main__":
    client = docker.from_env()
    print("Docker client initialized successfully.")

    print("Building Docker image...")
    client.images.build(
        path=str(REPO),
        dockerfile=str(REPO / "tools" / "Dockerfile"),
        tag=IMAGE,
    )
    print(f"Docker image built successfully with tag '{IMAGE}'.")

    code_submission_dir, csv_submission_dir = prepare_submission_variants()

    run_pipeline(
        client,
        mode="code",
        submission_dir=code_submission_dir,
        ingestion_res=REPO / "ingestion_res_code",
        scoring_res=REPO / "scoring_res_code",
    )
    run_pipeline(
        client,
        mode="csv",
        submission_dir=csv_submission_dir,
        ingestion_res=REPO / "ingestion_res_csv",
        scoring_res=REPO / "scoring_res_csv",
    )

    print("Docker pipelines ran successfully.")
