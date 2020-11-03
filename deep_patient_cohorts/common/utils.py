import re
from typing import List

import pandas as pd

POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0

IS_A_CID = "116680003"


def reformat_icd_code(icd_code: str, is_diag: bool = True) -> str:
    """Put a period in the right place because the MIMIC-3 data files exclude them.
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


def get_snowmed_children_descriptions(
    relationships_filepath: str, description_filepath: str, destination_id: str
) -> List[str]:
    """Given a parent concept in Snowmed (destination_id), returns the textual description of all
    its children concepts (i.e. those that have a "is a" relationship with the parent concept).
    Requires the filepaths to the relationship (relationships_filepath) and
    description (description_filepath) files from a Snowmed release.
    """
    relationships = pd.read_csv(
        relationships_filepath,
        sep="\t",
        usecols=["sourceId", "destinationId", "relationshipGroup", "typeId"],
        dtype={"sourceId": str, "destinationId": str, "relationshipGroup": str, "typeId": str},
    )
    descriptions = pd.read_csv(
        description_filepath,
        sep="\t",
        usecols=["conceptId", "term"],
        dtype={"conceptId": str, "term": str},
    )

    children_ids = (
        relationships[
            (relationships["destinationId"] == destination_id)
            & (relationships["typeId"] == IS_A_CID)
        ]["sourceId"]
        .unique()
        .tolist()
    )
    children_descriptions = descriptions[descriptions["conceptId"].isin(children_ids)]["term"]

    def _postprocess(text: str) -> str:
        text = re.sub(r"(\(substance\)|,?\s*NOS)", "", text)
        return text.strip()

    children_descriptions = children_descriptions.apply(_postprocess)

    return children_descriptions.unique().tolist()
