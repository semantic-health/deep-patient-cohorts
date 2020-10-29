import re
from typing import Callable, Iterable, List, Union

import numpy as np
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from deep_patient_cohorts.common.utils import ABSTAIN, NEGATIVE, POSITIVE

spacy.prefer_gpu()


class NoisyLabeler:
    def __init__(
        self,
        spacy_model: str = "en_core_sci_md",
    ) -> None:
        # Setup ScispaCy with the UMLS linking and Sectionizer pipes.
        nlp = spacy.load(spacy_model, disable=["tagger", "parser"])
        self._nlp = nlp

        self.lfs = [self._chest_pain, self._ejection_fraction, self._heart_disease, self._st_elevation,
                   self._atherosclerosis,self._heart_failure, self._angina, self._abnormal_diagnostic_test,
                   self._correlated_procedures, self._common_heart_failure]

    def add(self, lf: Callable) -> None:
        # Bind lf to self before appending
        self.lfs.append(lf.__get__(self))

    def preprocess(self, texts: Union[str, List[str], Iterable[str]]) -> List[Doc]:
        return [doc for doc in tqdm(self._nlp.pipe(texts))]

    def __call__(self, texts: Union[str, Iterable[Union[str, Doc]]]) -> np.ndarray:
        """Return an array, where the rows represent each text in `texts` and the columns contain
        the noisy labels produced by the labelling functions (LFs) in `self.lfs`."""
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

    def _chest_pain(self, texts: Iterable[Doc]) -> List[int]:
        return [POSITIVE if "chest pain" in text.text.lower() else ABSTAIN for text in texts]

    def _ejection_fraction(self, texts: Iterable[Doc]) -> List[int]:
        """Votes `POSITIVE` if a low ejection fraction is mentioned. Otherwise votes ABSTAIN.
        
        regex unit tests can be found here: https://regex101.com/r/mCw0b6/1
        """
        upper_bound = 35  # ejection fractions under which to vote POSITIVE, exclusively.
        pattern = re.compile(r"(ejection fraction|L?V?EF)[\s\w:<>=]+(\d\d)[\d-]*%?", re.IGNORECASE)
        matches = [pattern.findall(text.text) for text in texts]
        return [
            ABSTAIN if not match else POSITIVE if int(match[-1][-1]) < upper_bound else ABSTAIN
            for match in matches
        ]

        # heart_disease
    def _heart_disease(self, texts: List[str]) -> List[int]:
        return [POSITIVE if "heart disease" in text.text.lower() else ABSTAIN for text in texts]


    # st elevation LF
    def _st_elevation(self, texts: List[str]) -> List[int]:
        search_list = ["stemi", "st elevation", "st elevation mi"]
        return [POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN for text in texts]

    # atherosclerosis
    def _atherosclerosis(self, texts: List[str]) -> List[int]:
        search_list = ["atherosclerosis", "arteriosclerosis", "atherosclerotic", "arterial sclerosis", "artherosclerosis", "atherosclerotic disease"]
        return [POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN for text in texts]
    '''
    # heart_attack -- This labelling function is gives 0% accuracy thats why I commented it out

    def heart_attack(self, texts: List[str]) -> List[int]:
        search_list = ["myocardial infarcation", "mi", "ischemic heart disease", "cardiac arrest", "coronary infarction", "asystole", "cardiopulmonary arrest", "coronary thrombosis", "heart arrest", "heart attack", "heart stoppage"]
        return [POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN for text in texts]
    '''
    # heart_failure
    def _heart_failure(self, texts: List[str]) -> List[int]:
        search_list = ["congestive heart failure", "decomensated heart failure", "chf", "left-side heart failure", "right-sided heart failure"]
        return [POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN for text in texts]


    # angina LF

    def _angina(self, texts: List[str]) -> List[int]:
        search_list_1 = ["stable", "unstable",  "variant"]
        search_list_2 = ["angina", "chest_pain", "angina pectoris"]

        return [POSITIVE if (any([x in text.text.lower() for x in search_list_1])) and (any([x in text.text.lower() for x in search_list_2]))  else ABSTAIN for text in texts]


    # abnormal diagnostic tests results LF
    def _abnormal_diagnostic_test(self, texts: List[str]) -> List[int]:
        search_list_1 = ["abnormal", "concerning"]
        search_list_2 = ["ecg", "echo", "echocarrdiogram"]

        return [POSITIVE if (any([x in text.text.lower() for x in search_list_1])) and (any([x in text.text.lower() for x in search_list_2]))  else ABSTAIN for text in texts]

    # corelated procedures LF
    def _correlated_procedures(self, texts:List[str]) -> List[int]:
        search_list = ["coronary", "cardiac cath", "cardiac stent", "catheter", "catheterization", "stenting", "angioplasty",
                      "percutaneous coronary intervention","pci"]

        return [POSITIVE if any([x in text.text.lower() for x in search_list]) else ABSTAIN for text in texts]

    #Common symptom of heart failure

    def _common_heart_failure(self, texts:List[str]) -> List[int]:
        pattern = re.compile(r"(swelling|edema|puffiness)[\s\w:<>=]+(in)?[\s\w:<>=]+(left|right|l|r)?[\s\w:<>=]+(ankle|leg|feet|foot|ankles|legs)", re.IGNORECASE)
        matches = [pattern.findall(text.text) for text in texts]
        #print(matches)
        return [ABSTAIN if not match else POSITIVE for match in matches]



