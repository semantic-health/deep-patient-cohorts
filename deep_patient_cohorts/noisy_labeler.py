from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from deep_patient_cohorts.utils.common import reformat_icd_code

POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0


class NoisyLabeler:
    def __init__(
        self,
        descriptions: str,
        whitelist: Optional[Iterable[str]] = None,
        spacy_model: str = "en_core_sci_sm",
    ) -> None:
        self.nlp = spacy.load(spacy_model)
        # Load the ICD code descriptions provided by MIMIC-III and obtained here:
        # (https://physionet.org/content/mimiciii-demo/1.4/D_ICD_DIAGNOSES.csv)
        # This file does not contain properly formatted ICD codes, so we reformat.
        self._descriptions = (
            pd.read_csv(descriptions, converters={"ICD9_CODE": reformat_icd_code})
            .drop(["ROW_ID"], axis=1)
            .set_index("ICD9_CODE")
        )
        # If a whitelist is provided, restrict ourselves to the intersection of codes.
        whitelist = set() if whitelist is None else set(whitelist)
        self._icd_codes = set(self._descriptions.index) & whitelist

        self._lfs = [self._exact_match]

    def _exact_match(self, icd_code: str, documents: List[str]) -> List[int]:
        # Get the long title, break it up into tokens.
        # If any of the terms are in document, return POSITIVE, else return ABSTAIN.
        terms = ",".join(
            self._descriptions.loc[icd_code, "SHORT_TITLE":"LONG_TITLE"].tolist()
        ).split(",")
        matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        patterns = list(self.nlp.pipe(terms, disable=["parser", "ner"]))
        matcher.add(icd_code, None, *patterns)
        noisy_labels = [
            POSITIVE if matcher(doc) else ABSTAIN
            for doc in self.nlp.pipe(documents, disable=["parser", "ner"])
        ]
        return noisy_labels

    def __call__(self, documents: Union[str, List[str], Iterable[str]]) -> Dict[str, np.ndarray]:
        if isinstance(documents, str):
            documents = [documents]

        # For each ICD, we need to fill up an array of ( no. of documents X no. of lfs )
        noisy_labels = {icd: [] for icd in self._icd_codes}
        for code, labels in tqdm(noisy_labels.items()):
            for lf in self._lfs:
                labels.append(lf(code, documents))
        noisy_labels = {
            icd: np.asarray(labels, dtype=np.int8).T for icd, labels in noisy_labels.items()
        }
        return noisy_labels
