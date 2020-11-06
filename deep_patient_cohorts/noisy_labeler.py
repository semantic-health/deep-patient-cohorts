import re
from typing import Callable, Iterable, List, Union

import numpy as np
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from functools import partial

from deep_patient_cohorts.common.utils import (
    ABSTAIN,
    NEGATIVE,
    POSITIVE,
    HEART_DISEASES,
    CARDIAC_DRUGS,
)

spacy.prefer_gpu()


class NoisyLabeler:
    def __init__(
        self,
        spacy_model: str = "en_ner_bc5cdr_md",
    ) -> None:
        # Setup ScispaCy with the UMLS linking and Sectionizer pipes.
        nlp = spacy.load(spacy_model, disable=["tagger", "parser"])
        self._nlp = nlp

        self.lfs = [
            self._ejection_fraction,
            partial(self._exact_term_match, terms=HEART_DISEASES, threshold=2),
            partial(
                self._exact_term_match,
                terms=CARDIAC_DRUGS,
                threshold=1,
                negative_if_none=False,
            ),
            self._st_elevation,
            self._atherosclerosis,
            self._heart_failure,
            self._angina,
            self._abnormal_diagnostic_test,
            self._correlated_procedures,
            self._common_heart_failure,
        ]

    def add(self, lf: Callable[..., List[int]]) -> None:
        """Add labelling function `lf` to this instance."""
        # Bind lf to self before appending
        self.lfs.append(lf.__get__(self))  # type: ignore

    def preprocess(self, texts: Union[str, List[str], Iterable[str]]) -> List[Doc]:
        """Returns a list of `spacy.tokens.Doc` objects, one for each item in `texts`."""
        return [doc for doc in tqdm(self._nlp.pipe(texts))]

    def __call__(self, texts: Union[str, List[Union[str, Doc]]]) -> np.ndarray:
        """Return an array, where the rows represent each text in `texts` and the columns contain
        the noisy labels produced by the labelling functions (LFs) in `self.lfs`.
        """
        if not self.lfs:
            raise ValueError("At least one labelling function must be provided. self.lfs is empty.")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts[0], Doc):
            texts = self.preprocess(texts)

        noisy_labels = []
        for lf in tqdm(self.lfs):
            noisy_labels.append(lf(texts))
        return np.stack(noisy_labels, axis=-1)

    @staticmethod
    def accuracy(noisy_labels: np.ndarray, gold_labels: np.ndarray) -> None:
        """Print the accuracy and abstain rate of each labelling function in `noisy_labels` based
        on the given `gold_labels`.
        """
        m = noisy_labels.shape[1]

        for i in range(m):
            num_predictions = np.sum(noisy_labels[:, i] != ABSTAIN)
            accuracy = (
                np.sum(noisy_labels[:, i] == gold_labels) / num_predictions
                if num_predictions
                else 1
            )
            abstain_rate = np.sum(noisy_labels[:, i] == ABSTAIN) / gold_labels.shape[0]

            print(
                f"LF {i+1}: Accuracy {int(accuracy * 100)}%, Abstain rate {int((abstain_rate) * 100)}%"
            )

    def _ejection_fraction(self, texts: Iterable[Doc]) -> List[int]:
        """Votes POSITIVE if a low ejection fraction is mentioned. Otherwise votes ABSTAIN.

        Regex unit tests can be found here: https://regex101.com/r/mCw0b6/1
        """
        upper_bound = 35  # ejection fractions under which to vote POSITIVE, exclusively.
        pattern = re.compile(r"(ejection fraction|L?V?EF)[\s\w:<>=]+(\d\d)[\d-]*%?", re.IGNORECASE)
        matches = [pattern.findall(text.text) for text in texts]
        return [
            ABSTAIN if not match else POSITIVE if int(match[-1][-1]) < upper_bound else ABSTAIN
            for match in matches
        ]

    def _exact_term_match(
        self,
        texts: Iterable[Doc],
        terms: Iterable[str],
        threshold: int = 1,
        negative_if_none: bool = True,
    ):
        """Votes POSITIVE if there are `threshold` number of instances of `terms` for each `Doc`
        in `texts`. If `negative_if_none`, votes NEGATIVE if there are no matches. Otherwise votes
        ABSTAIN.
        """

        noisy_labels = []
        for text in texts:
            num_mentions = len(re.findall(r"|".join(terms), text.text, re.IGNORECASE))
            if num_mentions == 0 and negative_if_none:
                noisy_labels.append(NEGATIVE)
            elif num_mentions >= threshold:
                noisy_labels.append(POSITIVE)
            else:
                noisy_labels.append(ABSTAIN)
        return noisy_labels

    def _st_elevation(self, texts: List[Doc]) -> List[int]:
        search_list = ["stemi", "st elevation", "st elevation mi"]
        return [
            POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN
            for text in texts
        ]

    def _atherosclerosis(self, texts: Iterable[Doc]) -> List[int]:
        search_list = [
            "atherosclerosis",
            "arteriosclerosis",
            "atherosclerotic",
            "arterial sclerosis",
            "artherosclerosis",
            "atherosclerotic disease",
        ]
        return [
            POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN
            for text in texts
        ]

    def _heart_failure(self, texts: Iterable[Doc]) -> List[int]:
        search_list = [
            "congestive heart failure",
            "decomensated heart failure",
            "chf",
            "left-side heart failure",
            "right-sided heart failure",
        ]
        return [
            POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN
            for text in texts
        ]

    def _angina(self, texts: Iterable[Doc]) -> List[int]:
        search_list_1 = ["stable", "unstable", "variant"]
        search_list_2 = ["angina", "chest_pain", "angina pectoris"]

        return [
            POSITIVE
            if (any([x in text.text.lower() for x in search_list_1]))
            and (any([x in text.text.lower() for x in search_list_2]))
            else ABSTAIN
            for text in texts
        ]

    def _abnormal_diagnostic_test(self, texts: Iterable[Doc]) -> List[int]:
        search_list_1 = ["abnormal", "concerning"]
        search_list_2 = ["ecg", "echo", "echocarrdiogram"]

        return [
            POSITIVE
            if (any([x in text.text.lower() for x in search_list_1]))
            and (any([x in text.text.lower() for x in search_list_2]))
            else ABSTAIN
            for text in texts
        ]

    def _correlated_procedures(self, texts: Iterable[Doc]) -> List[int]:
        search_list = [
            "coronary",
            "cardiac cath",
            "cardiac stent",
            "catheter",
            "catheterization",
            "stenting",
            "angioplasty",
            "percutaneous coronary intervention",
            "pci",
        ]

        return [
            POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN
            for text in texts
        ]

    def _common_heart_failure(self, texts: Iterable[Doc]) -> List[int]:
        pattern = re.compile(
            (
                r"(swelling|edema|puffiness)[\s\w:<>=]+(in)?[\s\w:<>=]+(left|right|l|r)?[\s\w:<>=]"
                r"+(ankle|leg|feet|foot|ankles|legs)"
            ),
            re.IGNORECASE,
        )
        matches = [pattern.findall(text.text) for text in texts]
        return [ABSTAIN if not match else POSITIVE for match in matches]
