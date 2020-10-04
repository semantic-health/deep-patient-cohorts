from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import spacy
from sklearn.preprocessing import MultiLabelBinarizer
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from deep_patient_cohorts.classifiers.age import AgeClassifier
from deep_patient_cohorts.classifiers.sex import SexClassifier
from deep_patient_cohorts.common.utils import CHILDBIRTH_IDS, MENSTRUATION_AND_MENOPAUSE_IDS

spacy.prefer_gpu()


POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0


class NoisyLabeler:
    def __init__(
        self,
        labels: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        spacy_model: str = "en_core_sci_sm",
    ) -> None:
        self._labels = labels
        self._descriptions = descriptions
        self._nlp = spacy.load(spacy_model)

        self._classifiers = {
            "age": AgeClassifier(),
            "sex": SexClassifier(),
        }
        self._lfs = [
            self._phrase_match,
            self._negate_childbirth,
            self._negate_menstruation_and_menopause,
        ]

    def __call__(self, texts: Union[str, List[str], Iterable[str]]) -> Dict[str, np.ndarray]:

        if isinstance(texts, str):
            texts = [texts]

        sex = self._classifiers["sex"](texts)
        age = self._classifiers["age"](texts)

        # For each ICD, we need to fill up an array of ( no. of documents X no. of lfs )
        noisy_labels = {label: [] for label in self._labels}
        for label, noisy_label in tqdm(noisy_labels.items()):
            for lf in self._lfs:
                noisy_label.append(lf(label, texts, sex=sex, age=age))
        noisy_labels = {
            label: np.asarray(noisy_label, dtype=np.int8).T
            for label, noisy_label in noisy_labels.items()
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

    def _phrase_match(self, label: str, texts: List[str], **kwargs) -> List[int]:
        noisy_labels = [ABSTAIN] * len(texts)
        description = self._descriptions.get(label)
        if description:
            matcher = PhraseMatcher(self._nlp.tokenizer.vocab, attr="LOWER")
            patterns = list(self._nlp.tokenizer.pipe(description))
            matcher.add(label, patterns)
            docs = self._nlp.tokenizer.pipe(texts)
            noisy_labels = list(POSITIVE if matcher(doc) else ABSTAIN for doc in docs)
        return noisy_labels

    def _negate_childbirth(self, label: str, texts: List[str], **kwargs) -> List[str]:
        noisy_labels = np.array([ABSTAIN] * len(texts))
        if label in CHILDBIRTH_IDS:
            sex = kwargs.get("sex")
            age = kwargs.get("age")
            noisy_labels[np.logical_or(sex == 0, np.logical_and(age >= 75, age != -1))] = NEGATIVE
        return noisy_labels.tolist()

    def _negate_menstruation_and_menopause(
        self, label: str, texts: List[str], **kwargs
    ) -> List[str]:
        noisy_labels = np.array([ABSTAIN] * len(texts))
        if label in MENSTRUATION_AND_MENOPAUSE_IDS:
            sex = kwargs.get("sex")
            noisy_labels[sex == 0] = NEGATIVE
        return noisy_labels.tolist()
