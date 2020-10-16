from typing import Callable, Iterable, List, Union

import numpy as np
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from tqdm import tqdm

from deep_patient_cohorts.common.utils import ABSTAIN, NEGATIVE, POSITIVE

spacy.prefer_gpu()


class NoisyLabeler:
    def __init__(
        self,
        spacy_model: str = "en_core_sci_md",
    ) -> None:
        # Setup ScispaCy with the UMLS linking pipe.
        nlp = spacy.load(spacy_model)
        nlp.add_pipe(AbbreviationDetector(nlp))
        nlp.add_pipe(EntityLinker(resolve_abbreviations=True, name="umls"))
        self._nlp = nlp

        self.lfs = [
            self._chest_pain,
        ]

    def add(self, lf: Callable) -> None:
        # Bind lf to self before appending
        self.lfs.append(lf.__get__(self))

    def __call__(self, texts: Union[str, List[str], Iterable[str]]) -> np.ndarray:
        """Return an array, where the rows represent each text in `texts` and the columns contain
        the noisy labels produced by the labelling functions (LFs) in `self._lfs`."""
        if not self.lfs:
            raise ValueError("At least one labelling function must be provided. self.lfs is empty.")
        if isinstance(texts, str):
            texts = [texts]

        noisy_labels = []
        for lf in tqdm(self.lfs):
            noisy_labels.append(lf(texts))
        return np.stack(noisy_labels, axis=-1)

    @staticmethod
    def accuracy(noisy_labels: np.ndarray, gold_labels: np.ndarray) -> None:
        """Print the accuracy and abstain rate of each labelling function in `noisy_labels`,
        based on the given `gold_labels`."""
        m = noisy_labels.shape[1]

        for i in range(m):
            num_predictions = np.sum(noisy_labels[:, i] != ABSTAIN)
            if num_predictions != 0:
                accuracy = np.sum(noisy_labels[:, i] == gold_labels) / num_predictions
            else:
                accuracy = 1
            abstain_rate = np.sum(noisy_labels[:, i] == ABSTAIN) / gold_labels.shape[0]

            print(
                f"LF {i}: Accuracy {int(accuracy * 100)}%, Abstain rate {int((abstain_rate) * 100)}%"
            )

    def _chest_pain(self, texts: List[str]) -> List[int]:
        return [POSITIVE if "chest pain" in text.lower() else ABSTAIN for text in texts]
