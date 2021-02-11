import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0

IS_A_CID = "116680003"

TERM_LISTS = Path(__file__).parent / "term_lists"
# The textual descriptions of every child of the Snowmed Concept "Heart disease" (56265001)
HEART_DISEASES: set = set((TERM_LISTS / "heart_disease.txt").read_text().split("\n"))
# The textual descriptions of every child of the Snowmed Concept "Cardiovascular agents" (373247007)
CARDIAC_DRUGS: set = set((TERM_LISTS / "cardiac_drugs.txt").read_text().split("\n"))
# The textual descriptions of every child of the Snowmed Concept "Vascular sclerosis" (107671003)
VASCULAR_SCLEROSIS: set = set((TERM_LISTS / "vascular_sclerosis.txt").read_text().split("\n"))
# The textual descriptions of every child of the Snowmed Concept "Cardiovascular investigation" (276341003)
CARDIAC_PROCEDURES: set = set((TERM_LISTS / "cardiac_procedures.txt").read_text().split("\n"))


def reformat_icd_code(icd_code: str, is_diag: bool = True) -> str:
    """Put a period in the right place because the MIMIC-III data files exclude them.
    Generally, procedure ICD codes have dots after the first two digits, while diagnosis
    ICD codes have dots after the first three digits.

    Adopted from: https://github.com/jamesmullenbach/caml-mimic
    """
    icd_code = "".join(icd_code.split("."))
    if is_diag:
        if icd_code.startswith("E"):
            if len(icd_code) > 4:
                icd_code = icd_code[:4] + "." + icd_code[4:]
        else:
            if len(icd_code) > 3:
                icd_code = icd_code[:3] + "." + icd_code[3:]
    else:
        icd_code = icd_code[:2] + "." + icd_code[2:]
    return icd_code


def create_term_list(
    destination_id: str,
    description_filepath: str,
    relationships_filepath: str,
    output_filepath: str,
    num_levels: Optional[int] = None,
    max_length: Optional[int] = None,
) -> None:
    """Given a parent concept in Snowmed (`destination_id`), creates a text file at
    `output_filepath` containing the textual descriptions of all its children concepts, i.e. those
    that have a "is a" relationship with the parent concept. Requires the filepaths to the
    relationship (`relationships_filepath`) and description (`description_filepath`) files from a
    Snowmed release.

    # Parameters:

    destination_id : `str`, required.
        The Snowmed ID of the parent concept.
    description_filepath : `str`, required.
        The filepath to the decription file from a Snowmed release.
    relationships_filepath : `str`, required.
        The filepath to the relationships file from a Snowmed release.
    output_filepath : `str`, required.
        The filepath to save the retrieved term list.
    num_levels : `int`, optional.
        The number of levels to traverse in the Snowmed tree when retrieving child concepts.
        A sensible default is 2.
    max_length : `int`, optional.
        If not `None`, only terms with a number of whitespace tokens equal to or less than this
        number are retained. A sensible default is 4.
    """
    descriptions = pd.read_csv(
        description_filepath,
        sep="\t",
        usecols=["conceptId", "term"],
        dtype={"conceptId": str, "term": str},
    )
    relationships = pd.read_csv(
        relationships_filepath,
        sep="\t",
        usecols=["sourceId", "destinationId", "typeId"],
        dtype={"sourceId": str, "destinationId": str, "typeId": str},
    )
    children_ids = (
        relationships[
            (relationships["destinationId"] == destination_id)
            & (relationships["typeId"] == IS_A_CID)
        ]["sourceId"]
        .unique()
        .tolist()
    )
    previous_children_ids = children_ids[:]

    current_level = 0
    while True:
        current_children_ids = []
        for child_id in tqdm(previous_children_ids):
            current_children_ids.extend(
                relationships[
                    (relationships["destinationId"] == child_id)
                    & (relationships["typeId"] == IS_A_CID)
                ]["sourceId"]
                .unique()
                .tolist()
            )
        current_level += 1
        previous_children_ids = current_children_ids[:]
        children_ids.extend(current_children_ids)
        if not previous_children_ids or (num_levels and current_level == num_levels):
            break

    def _postprocess(text: str) -> str:
        # Remove irrelevant text in textual description. Unfortunately there doesn't
        # appear to be a pattern, so we have to target these one-by-one.
        text = re.sub(
            r"\((finding|substance|disorder|morphologic\sabnormality|(medicinal\s)?product|\w-\d+|"
            r"procedure|regime/therapy)\)|-\s*chemical|-class of antibiotic-|\[Ambiguous\]|,?\s*NOS",
            "",
            text,
        )
        # Remove whitespace, newlines and tabs.
        text = " ".join(text.strip().split())
        return text

    children_descriptions = (
        # Include the description of the parent concept.
        descriptions[descriptions["conceptId"].isin(set(children_ids + [destination_id]))]["term"]
        .apply(_postprocess)
        .unique()
        .tolist()
    )
    if max_length:
        children_descriptions = [
            descr for descr in children_descriptions if len(descr.split()) <= max_length
        ]

    output_filepath: Path = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    output_filepath.write_text("\n".join(children_descriptions))
