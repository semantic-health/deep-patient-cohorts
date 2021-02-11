import numpy as np
import pytest
from deep_patient_cohorts.common.utils import ABSTAIN, NEGATIVE, POSITIVE
from deep_patient_cohorts.noisy_labeler import NoisyLabeler


class TestNoisyLabeler:
    def test_fit_lfs_value_error(self, noisy_labeler: NoisyLabeler) -> None:
        with pytest.raises(ValueError):
            noisy_labeler.lfs = []
            noisy_labeler.fit_lfs(texts=[])

    def test_predict_value_error(self, noisy_labeler: NoisyLabeler) -> None:
        with pytest.raises(ValueError):
            noisy_labeler.label_model = None
            noisy_labeler.predict(texts=[])

    def test_accuracy(self) -> None:
        noisy_labels = np.asarray([[POSITIVE, ABSTAIN, NEGATIVE], [NEGATIVE, ABSTAIN, ABSTAIN]])
        gold_labels = np.asarray([POSITIVE, NEGATIVE])

        expected = ([1.0, 1.0, 0.0], [0.0, 1.0, 0.5])
        actual = NoisyLabeler.accuracy(noisy_labels, gold_labels)
        assert actual == expected

    def test_accuracy_value_error_noisy_labels(self) -> None:
        noisy_labels = np.asarray([[[POSITIVE]]])
        gold_labels = np.asarray([POSITIVE])
        with pytest.raises(ValueError):
            _ = NoisyLabeler.accuracy(noisy_labels, gold_labels)

    def test_accuracy_value_error_gold_labels(self) -> None:
        noisy_labels = np.asarray([[POSITIVE]])
        gold_labels = np.asarray([[[POSITIVE]]])
        with pytest.raises(ValueError):
            _ = NoisyLabeler.accuracy(noisy_labels, gold_labels)
