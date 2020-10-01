import random
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import csv

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main(
    noteevents_filepath: str = typer.Argument(
        ..., help="Filepath to the NOTEEVENTS.csv from MIMIC-III."
    ),
    patients_filepath: str = typer.Argument(
        ..., help="Filepath to the PATIENTS.csv from MIMIC-III."
    ),
    output_filepath: str = typer.Argument(
        "./train.tsv", help="Filepath to save the tab-seperated labelled data file."
    ),
    num_examples: int = typer.Option(
        20000, help="The number of examples to randomly sample to form the train set."
    ),
):
    typer.secho(
        f"Loading data from {noteevents_filepath}...",
        bold=True,
    )
    noteevents = pd.read_csv(
        noteevents_filepath,
        header=0,
        index_col="SUBJECT_ID",
        usecols=["SUBJECT_ID", "TEXT", "CATEGORY"],
        # Remove whitespace, newlines and tabs from TEXT column data.
        converters={"TEXT": lambda text: " ".join(text.strip().split())},
    )
    # Filter out anything that isn't a discharge summary
    noteevents = noteevents[noteevents["CATEGORY"] == "Discharge summary"].drop("CATEGORY", axis=1)

    typer.secho(
        f"Loading data from {patients_filepath}...",
        bold=True,
    )
    patients = pd.read_csv(
        patients_filepath, header=0, index_col="SUBJECT_ID", usecols=["SUBJECT_ID", "GENDER"]
    )

    df = noteevents.join(patients)

    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    # Sample exactly 1:1 ratio of females to males
    f = (df[df["GENDER"] == "F"]).sample(num_examples // 2, random_state=RANDOM_STATE)
    m = (df[df["GENDER"] == "M"]).sample(num_examples // 2, random_state=RANDOM_STATE)
    train_set = pd.concat([f, m]).sample(frac=1, random_state=RANDOM_STATE)
    train_set.to_csv(
        output_filepath,
        sep="\t",
        columns=["TEXT", "GENDER"],
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
    )
    typer.secho(
        f"Labelled data saved to {output_filepath.absolute()}.",
        bold=True,
    )


if __name__ == "__main__":
    typer.run(main)
