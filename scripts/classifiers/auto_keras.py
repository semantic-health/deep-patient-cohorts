from pathlib import Path
import autokeras as ak
import numpy as np
import pandas as pd
import random

import typer

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main(
    input_filepath: str = typer.Argument(..., help="Filepath to the TSV-formatted train dataset."),
    output_directory: str = typer.Argument(
        "./", help="Directory to save the best model found during the search."
    ),
    max_trials: int = typer.Option(
        100,
        help=(
            "The maximum number of different Keras Models to try."
            " The search may finish before reaching the max_trials."
        ),
    ),
):
    df = pd.read_csv(input_filepath, sep="\t", header=None, names=["text", "labels"])
    X = df["text"].values.astype(str)
    y = df["labels"].values

    # Hardcod
    clf = ak.TextClassifier(max_trials=max_trials, seed=RANDOM_STATE)
    clf.fit(X, y)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    clf.export_model(output_directory)
    typer.secho(
        f"Best model saved to {output_directory.absolute()}.", bold=True,
    )


if __name__ == "__main__":
    typer.run(main)
