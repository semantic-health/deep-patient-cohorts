from typing import List, Union

import autokeras as ak
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path


class SexClassifier:
    def __init__(self):
        self._model = load_model(
            Path(__file__).absolute().parents[2] / "data/classifiers/sex/model_autokeras",
            custom_objects=ak.CUSTOM_OBJECTS,
        )

    def __call__(self, texts: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """Returns an array of ints, 0 if the patient is male and 1 if the patient is female."""
        if isinstance(texts, str):
            texts = np.asarray([texts], dtype=str)
        elif isinstance(texts, list):
            texts = np.asarray(texts, dtype=str)
        return (self._model.predict(texts) < 0.5).ravel().astype(int)
