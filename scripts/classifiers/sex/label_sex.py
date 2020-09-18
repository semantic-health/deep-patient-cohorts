import json
import os
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main(
    input_filepath: str = typer.Argument(
        ..., help="Filepath to the JSON Lines formatted dataset to label."
    ),
    output_directory: str = typer.Argument(
        "./", help="Directory to save the labelled data file 'train.tsv'."
    ),
    num_examples: int = typer.Option(
        500,
        help=(
            "The number of examples from the input dataset to consider for labelling."
            " Some of these will be automatically labelled."
        ),
    ),
):
    texts = [
        json.loads(line)["text"] for line in Path(input_filepath).read_text().strip().split("\n")
    ]
    labels = ["f" if "sex: f" in text else "m" if "sex: m" in text else "u" for text in texts]
    test_size = num_examples / len(texts)
    _, texts, _, labels = train_test_split(
        texts, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
    )

    num_undefined_before = np.sum(np.asarray(labels) == "u")
    typer.secho(
        f"There are {num_undefined_before}/{len(labels)} examples with undefined sex to label.",
        bold=True,
    )
    _ = typer.confirm("Begin labelling?", abort=True)

    labelled = 0
    for i, label in enumerate(labels):
        if label != "u":
            continue
        print(texts[i])
        user_input = str(input(f"({labelled}/{num_undefined_before}) m/f/u?: ")).lower().strip()
        while user_input not in {"f", "m", "u"}:
            typer.secho('Please select "f" (female), "m" (male) or "u" (undefined)')
            user_input = str(input(f"({labelled}/{num_undefined_before}) m/f/u?: ")).lower().strip()
        labels[i] = user_input
        labelled += 1
        os.system("cls" if os.name == "nt" else "clear")

    num_undefined_after = np.sum(np.asarray(labels) == "u")
    typer.secho(
        (
            f"After labelling, {num_undefined_after}/{num_undefined_before} examples with undefined"
            " sex remain."
        ),
        bold=True,
    )

    output_filepath = Path(output_directory)
    output_filepath.mkdir(parents=True, exist_ok=True)
    output_filepath = output_filepath / "train.tsv"
    with open(output_filepath, "w") as f:
        for text, label in zip(texts, labels):
            f.write(f"{text}\t{label}\n")
    typer.secho(
        f"Labelled data saved to {output_filepath.absolute()}.", bold=True,
    )


if __name__ == "__main__":
    typer.run(main)
