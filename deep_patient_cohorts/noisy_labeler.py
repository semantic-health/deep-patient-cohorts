from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from deep_patient_cohorts.utils.common import reformat_icd_code

spacy.prefer_gpu()


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
        self._icd_codes = self._descriptions.index.tolist()
        if whitelist is not None:
            self._icd_codes = list(set(self._icd_codes) & set(whitelist))

        self._lfs = [self._phrase_match, self._negate_pregancy]

    def __call__(self, texts: Union[str, List[str], Iterable[str]]) -> Dict[str, np.ndarray]:

        if isinstance(texts, str):
            texts = [texts]
        # For each ICD, we need to fill up an array of ( no. of documents X no. of lfs )
        noisy_labels = {icd: [] for icd in self._icd_codes}
        for code, labels in tqdm(noisy_labels.items()):
            for lf in self._lfs:
                labels.append(lf(code, texts))
        noisy_labels = {
            icd: np.asarray(labels, dtype=np.int8).T for icd, labels in noisy_labels.items()
        }

        return noisy_labels

    def _phrase_match(self, icd_code: str, texts: List[str]) -> List[int]:
        # Get the long title, break it up into tokens.
        # If any of the terms are in document, return POSITIVE, else return ABSTAIN.
        descriptions = ",".join(
            self._descriptions.loc[icd_code, "SHORT_TITLE":"LONG_TITLE"].tolist()
        ).split(",")
        matcher = PhraseMatcher(self.nlp.tokenizer.vocab, attr="LOWER")
        patterns = list(self.nlp.tokenizer.pipe(descriptions))
        matcher.add(icd_code, patterns)
        docs = self.nlp.tokenizer.pipe(texts)
        noisy_labels = list(POSITIVE if matcher(doc) else ABSTAIN for doc in docs)
        return noisy_labels

    def _negate_pregancy(self, icd_code: str, texts: List[str]) -> List[int]:
        pregnancy_icd_code = self._descriptions[
            self._descriptions.LONG_TITLE.str.contains("pregnancy", case=False)
        ].index.tolist()
        keyword = "sex: m"
        if icd_code in pregnancy_icd_code:
            noisy_labels = []
            for text in texts:
                if keyword in text:
                    noisy_labels.append(NEGATIVE)
                else:
                    noisy_labels.append(ABSTAIN)
        else:
            noisy_labels = [ABSTAIN] * len(texts)
        return noisy_labels
