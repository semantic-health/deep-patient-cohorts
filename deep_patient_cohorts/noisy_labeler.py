from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import MultiLabelBinarizer
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from deep_patient_cohorts.classifiers.age import AgeClassifier
from deep_patient_cohorts.classifiers.sex import SexClassifier
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
        self._class_descriptions = (
            pd.read_csv(descriptions, converters={"ICD9_CODE": reformat_icd_code})
            .drop(["ROW_ID"], axis=1)
            .set_index("ICD9_CODE")
        )
        # If a whitelist is provided, restrict ourselves to the intersection of codes.
        self._class_labels = self._class_descriptions.index.tolist()
        if whitelist is not None:
            self._class_labels = list(set(self._class_labels) & set(whitelist))

        self._classifiers = {
            "age": AgeClassifier(),
            "sex": SexClassifier(),
        }
        self._lfs = [self._phrase_match, self._negate_childbirth]

    def __call__(self, texts: Union[str, List[str], Iterable[str]]) -> Dict[str, np.ndarray]:

        if isinstance(texts, str):
            texts = [texts]
        # For each ICD, we need to fill up an array of ( no. of documents X no. of lfs )
        noisy_labels = {class_label: [] for class_label in self._class_labels}
        for class_label, noisy_label in tqdm(noisy_labels.items()):
            for lf in self._lfs:
                noisy_label.append(lf(class_label, texts))
        noisy_labels = {
            class_label: np.asarray(noisy_label, dtype=np.int8).T
            for class_label, noisy_label in noisy_labels.items()
        }
        return noisy_labels

    @staticmethod
    def accuracy(
        noisy_labels: Dict[str, np.ndarray], gold_labels: List[List[str]]
    ) -> Tuple[List[float], List[float]]:
        """Return the accuracy and abstain rate of each labelling function in `noisy_labels`,
        based on the given `gold_labels`."""
        # Binarize the labels, setting the negative (absence) of a class to
        # -1 to match FlyingSquids convention.
        mlb = MultiLabelBinarizer()
        gold_labels = mlb.fit_transform(gold_labels)
        gold_labels = np.where(
            gold_labels == 0, -1 * np.ones_like(gold_labels), np.ones_like(gold_labels)
        )

        # (no. of examples, no. of lfs)
        n, m = list(noisy_labels.values())[0].shape

        accuracy = [[] for _ in range(m)]
        abstain_rate = [[] for _ in range(m)]
        for i, class_ in enumerate(mlb.classes_):
            for j in range(m):
                if class_ in noisy_labels:
                    num_predictions = np.sum(noisy_labels[class_][:, j] != 0)
                    if num_predictions != 0:
                        accuracy[j].append(
                            np.sum(noisy_labels[class_][:, j] == gold_labels[:, i])
                            / num_predictions
                        )
                    abstain_rate[j].append(np.sum(noisy_labels[class_][:, j] == 0) / n)
                else:
                    abstain_rate[j].append(1)

        return accuracy, abstain_rate

    def _phrase_match(self, class_label: str, texts: List[str]) -> List[int]:
        # Get the long title, break it up into tokens.
        # If any of the terms are in document, return POSITIVE, else return ABSTAIN.
        descriptions = ",".join(
            self._class_descriptions.loc[class_label, "SHORT_TITLE":"LONG_TITLE"].tolist()
        ).split(",")
        matcher = PhraseMatcher(self.nlp.tokenizer.vocab, attr="LOWER")
        patterns = list(self.nlp.tokenizer.pipe(descriptions))
        matcher.add(class_label, patterns)
        docs = self.nlp.tokenizer.pipe(texts)
        noisy_labels = list(POSITIVE if matcher(doc) else ABSTAIN for doc in docs)
        return noisy_labels

    def _negate_childbirth(self, class_label: str, texts: List[str]) -> List[int]:
        # Determine if the current ICD code is pregnancy related
        descriptions = ",".join(
            self._class_descriptions.loc[class_label, "SHORT_TITLE":"LONG_TITLE"].tolist()
        ).split(",")
        matcher = PhraseMatcher(self.nlp.tokenizer.vocab, attr="LOWER")
        patterns = list(
            self.nlp.tokenizer.pipe(
                [
                    "birth",
                    "childbirth",
                    "pregnant",
                    "pregnancy",
                    "gestation",
                    "labor",
                    "delivery",
                    "complicating labor and delivery",
                    "outcome of delivery",
                ]
            ),
        )
        matcher.add(class_label, patterns)
        docs = self.nlp.tokenizer.pipe(descriptions)
        pregnancy_icd_code = any(matcher(doc) for doc in docs)
        # Use pre-trained classifier to determine sex and age
        is_male = self._classifiers["sex"](texts)
        ages = self._classifiers["age"](texts)
        # If pregnancy related ICD code and patient is male or young, vote NEGATIVE, else ABSTAIN
        if pregnancy_icd_code:
            noisy_labels = [
                NEGATIVE if male or age >= 75 else ABSTAIN for male, age in zip(is_male, ages)
            ]
        else:
            noisy_labels = [ABSTAIN] * len(texts)
        return noisy_labels
