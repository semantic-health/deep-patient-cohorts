import itertools
import re
from functools import partial
from itertools import chain
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import spacy
from flyingsquid.label_model import LabelModel
from spacy.tokens import Doc
from tqdm import tqdm

from deep_patient_cohorts.common.utils import (
    ABSTAIN,
    CARDIAC_DRUGS,
    CARDIAC_PROCEDURES,
    HEART_DISEASES,
    NEGATIVE,
    POSITIVE,
    VASCULAR_SCLEROSIS,
)

spacy.prefer_gpu()


class NoisyLabeler:
    def __init__(self, spacy_model: str = "en_ner_bc5cdr_md") -> None:
        # Setup ScispaCy with the UMLS linking and Sectionizer pipes.
        nlp = spacy.load(spacy_model, disable=["tagger", "parser"])
        self._nlp = nlp

        self.lfs = [
            partial(self._exact_term_match, terms=HEART_DISEASES, threshold=6),
            partial(self._exact_term_match, terms=CARDIAC_DRUGS, threshold=8),
            partial(
                self._exact_term_match,
                terms=VASCULAR_SCLEROSIS,
                threshold=2,
                negative_if_none=False,
            ),
            partial(self._exact_term_match, terms=CARDIAC_PROCEDURES, threshold=3),
            self._ejection_fraction,
            self._st_elevation,
            self._heart_failure,
            self._abnormal_diagnostic_test,
            self._correlated_procedures,
        ]
        self.lfs = [
            partial(
                self._exact_term_match,
                terms=CARDIAC_PROCEDURES,
                threshold=2,
                negative_if_none=False,
            )
        ]
        self.label_model: LabelModel = None

    def add(self, lf: Callable) -> None:
        # Bind lf to self before appending
        self.lfs.append(lf.__get__(self))  # type: ignore

    def preprocess(self, texts: Union[str, Iterable[Doc], Iterable[str]]) -> List[Doc]:
        return [doc for doc in tqdm(self._nlp.pipe(texts))]

    def fit_lfs(self, texts: Union[str, List[Union[str, Doc]]]) -> np.ndarray:
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
        for lf in tqdm(self.lfs, desc="Labelling data with LFs"):
            noisy_labels.append(lf(texts))
        return np.stack(noisy_labels, axis=-1)

    def fit_lm(self, noisy_labels: np.ndarray, gold_labels: np.ndarray) -> None:
        """Updates `self.label_model` with a `LabelModel` fitted on the combination of lfs in
        `self.lfs` that obtained the highest accuracy.
        """
        m = noisy_labels.shape[1]
        combinations = list(
            chain.from_iterable(itertools.combinations(list(range(m)), r=r) for r in range(3, m))
        )
        best_accuracy = 0
        best_combination = None
        best_label_model = None

        for combination in tqdm(combinations, desc="Fitting label models"):
            label_model = LabelModel(len(combination))
            label_model.fit(noisy_labels[:, combination])
            preds = label_model.predict(noisy_labels).reshape(gold_labels.shape)
            accuracy = np.sum(preds == gold_labels) / gold_labels.shape[0]

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = combination
                best_label_model = label_model

        print(
            f"The best label model used LFs {best_combination} and obtained an accuracy"
            f" of {best_accuracy*100:.2f}%"
        )
        self.label_model = best_label_model

    def fit(self, texts: Union[str, List[Union[str, Doc]]], gold_labels: np.ndarray) -> None:
        """Return an array, where the rows represent each text in `texts` and the columns contain
        the noisy labels produced by the labelling functions (LFs) in `self.lfs`.
        """
        noisy_labels = self.fit_lfs(texts)
        self.fit_lm(noisy_labels, gold_labels)

    def predict(self, texts: Union[str, List[Union[str, Doc]]]) -> np.ndarray:
        """Given some `texts`, returns the noisy labels produced by `self.label_model` as an array."""
        if self.label_model is None:
            raise ValueError(
                "self.label_model is None. You must first call self.fit_lm or self.fit"
            )
        noisy_labels = self.fit_lfs(texts)
        return self.label_model.predict(noisy_labels)

    @staticmethod
    def accuracy(
        noisy_labels: np.ndarray, gold_labels: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Returns a tuple of lists containing the accuracy and abstain rate of each labelling
        function in `noisy_labels`, based on the given `gold_labels`."""
        if len(noisy_labels.shape) != 2:
            raise ValueError(
                "Expected noisy_labels to be a 2-dimensional array."
                f" Got a {len(noisy_labels.shape)}-dimension array."
            )
        if len(gold_labels.shape) > 2:
            raise ValueError(
                "Expected gold_labels to be a 1-dimensional array."
                f" Got a {len(gold_labels.shape)}-dimension array."
            )

        m = noisy_labels.shape[1]
        gold_labels = gold_labels.squeeze(-1) if len(gold_labels.shape) == 2 else gold_labels

        accuracy, abstain_rate = [], []

        for i in range(m):
            num_predictions = np.sum(noisy_labels[:, i] != ABSTAIN)
            if num_predictions != 0:
                accuracy.append(np.sum(noisy_labels[:, i] == gold_labels) / num_predictions)
            else:
                accuracy.append(1)
            abstain_rate.append(np.sum(noisy_labels[:, i] == ABSTAIN) / gold_labels.shape[0])

            print(
                f"LF {i}: Accuracy {int(accuracy[-1] * 100)}%, Abstain rate {int((abstain_rate[-1]) * 100)}%"
            )
        return accuracy, abstain_rate

    def _exact_term_match(
        self,
        texts: Iterable[Doc],
        terms: Iterable[str],
        ignore_case: bool = True,
        threshold: int = 1,
        negative_if_none: bool = True,
        entity_class: str = None,
    ):
        """Votes POSITIVE if there are `mention_threshold` number of instances of `terms` for each
        `Doc` in `texts`. If `negative_if_none`, votes NEGATIVE if there are no matches. Otherwise
        votes ABSTAIN. To match on named entities identified by `spacy_model`, provide the class
        name with `entity_class`.
        """
        terms = [term.lower() if ignore_case else term for term in terms]
        noisy_labels = []

        for text in texts:
            if entity_class:
                ents = {
                    ent.text.lower() if ignore_case else ent.text
                    for ent in text.ents
                    if ent.label_ == entity_class
                }
                num_mentions = sum([term in ents for term in terms])
            else:
                text_to_match = text.text.lower() if ignore_case else text.text
                num_mentions = len(re.findall(r"|".join(terms), text_to_match))
            if num_mentions == 0 and negative_if_none:
                noisy_labels.append(NEGATIVE)
            elif num_mentions >= threshold:
                noisy_labels.append(POSITIVE)
            else:
                noisy_labels.append(ABSTAIN)
        return noisy_labels

    def _ejection_fraction(self, texts: Iterable[Doc], threshold: float = 30) -> List[int]:
        """Votes POSITIVE if an ejection fraction equal to or less than `threshold` is mentioned.
        Otherwise votes ABSTAIN.

        Regex unit tests can be found here: https://regex101.com/r/mCw0b6/1
        """
        pattern = re.compile(
            r"(?:ejection fraction|LVEF)[\s<>=:]+[\sa-zA-Z]*(\d\d)\D",
            re.IGNORECASE,
        )
        matches = [pattern.findall(text.text) for text in texts]

        return [POSITIVE if match and int(match[-1]) <= threshold else ABSTAIN for match in matches]

    def _st_elevation(self, texts: Iterable[Doc]) -> List[int]:
        search_list = ["stemi", "st elevation", "st elevation mi"]
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
