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
        "./",
        help=(
            "Directory to save the output generated during the search. The best model will be saved"
            " as 'output_directory/model_autokeras' or 'output_directory/model_autokeras.h5'"
        ),
    ),
    max_trials: int = typer.Option(
        1000,
        help=(
            "The maximum number of different Keras Models to try."
            " The search may finish before reaching the max_trials."
        ),
    ),
):
    df = pd.read_csv(input_filepath, sep="\t", header=None, names=["text", "labels"])
    X = df["text"].values.astype(str)
    y = df["labels"].values

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Hardcode max_model_size to the upper bound of the first models AutoKeras tries.
    clf = ak.TextClassifier(max_trials=max_trials, directory=output_directory, seed=RANDOM_STATE)
    clf.fit(X, y)

    model = clf.export_model()
    try:
        output_filepath = output_directory / "model_autokeras"
        model.save(output_filepath, save_format="tf")
    except ImportError:
        output_filepath = output_directory / "model_autokeras.h5"
        model.save(output_filepath)
    typer.secho(
        f"Best model saved to {output_filepath.absolute()}.",
        bold=True,
    )


if __name__ == "__main__":
    typer.run(main)
