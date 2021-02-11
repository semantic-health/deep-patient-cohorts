import pytest
from deep_patient_cohorts.noisy_labeler import NoisyLabeler


@pytest.fixture
def noisy_labeler():
    return NoisyLabeler()
