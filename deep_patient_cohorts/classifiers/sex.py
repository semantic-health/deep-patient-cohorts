from typing import List, Union

import autokeras as ak
import numpy as np
from tensorflow.keras.models import load_model


class SexClassifier:
    def __init__(self):
        self._model = load_model("../data/classifiers/sex/model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

    def __call__(self, texts: Union[str, List[str], np.ndarray]) -> str:
        if isinstance(texts, str):
            texts = np.asarray([texts], dtype=str)
        elif isinstance(texts, list):
            texts = np.asarray(texts, dtype=str)
        return self._model.predict(texts, dtype=str) < 0.5
