from pathlib import Path
import subprocess

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocess_feature.ipynb",
    "notebooks/03_mining_or_clustering.ipynb",
    "notebooks/04_modeling.ipynb",
    "notebooks/05_evaluation_report.ipynb",
]


def run_one(path: str):
    cmd = [
        "papermill",
        path,
        path,
        "-k",
        "python3",
    ]
    subprocess.run(cmd, check=True)


def main():
    for nb in NOTEBOOKS:
        if Path(nb).exists():
            run_one(nb)


if __name__ == "__main__":
    main()
