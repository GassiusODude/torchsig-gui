import pytest
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sl_main import DatasetMetadataIn, FullDatasetConfigRequest


def test_valid_dataset_metadata():
    metadata = DatasetMetadataIn(
        sample_rate=10000000,
        num_signals_min=1,
        num_signals_max=3,
        snr_db_min=0.0,
        snr_db_max=20.0,
        signal_duration_in_samples_min=100,
        signal_duration_in_samples_max=200,
        bandwidth_min=1000000,
        bandwidth_max=5000000,
        signal_center_freq_min=-100000,
        signal_center_freq_max=100000,
        frequency_min=-500000,
        frequency_max=500000
    )
    assert metadata.sample_rate == 10000000


@pytest.mark.parametrize(
    "kwargs,match_str",
    [
        ({"num_signals_min": 5, "num_signals_max": 2}, "num_signals_min"),
        ({"snr_db_min": 10.0, "snr_db_max": 5.0}, "snr_db_min"),
        ({"signal_duration_in_samples_min": 500, "signal_duration_in_samples_max": 200}, "signal_duration_in_samples_min"),
        ({"bandwidth_min": 3000000, "bandwidth_max": 1000000}, "bandwidth_min"),
        ({"signal_center_freq_min": 100, "signal_center_freq_max": -100}, "signal_center_freq_min"),
        ({"frequency_min": 1000, "frequency_max": 500}, "frequency_min"),
    ],
)
def test_min_greater_than_max_validations(kwargs, match_str):
    with pytest.raises(ValidationError, match=match_str):
        DatasetMetadataIn(**kwargs)


@pytest.mark.parametrize(
    "kwargs,match_str",
    [
        ({"sample_rate": 1000000, "bandwidth_min": 3000000, "bandwidth_max": 2000000}, "bandwidth_min.*must be less than or equal to bandwidth_max|bandwidth_max.*exceed sample_rate"),
        ({"sample_rate": 1000000, "bandwidth_min": 500000, "bandwidth_max": 2000000}, "bandwidth_max.*exceed sample_rate"),
    ],
)
def test_bandwidth_exceeds_sample_rate(kwargs, match_str):
    with pytest.raises(ValidationError, match=match_str):
        DatasetMetadataIn(**kwargs)


def test_empty_class_list_validation():
    metadata = DatasetMetadataIn()
    with pytest.raises(ValidationError, match="class_list cannot be empty"):
        FullDatasetConfigRequest(
            dataset_id="test_dataset",
            dataset_metadata=metadata,
            class_list=[]
        )


@pytest.mark.parametrize(
    "kwargs,match_str",
    [
        ({"sample_rate": 10000000, "signal_center_freq_min": -4000000, "bandwidth_max": 3000000}, "signal_center_freq_min - bandwidth_max / 2"),
        ({"sample_rate": 10000000, "signal_center_freq_max": 4000000, "bandwidth_max": 3000000}, "signal_center_freq_max \\+ bandwidth_max / 2"),
    ],
)
def test_spectral_occupancy_constraints(kwargs, match_str):
    with pytest.raises(ValidationError, match=match_str):
        DatasetMetadataIn(**kwargs)


@pytest.mark.parametrize(
    "sampling_mode,class_list,should_pass,match_str",
    [
        ("per_signal", ["ook", "bpsk"], True, None),
        ("per_family", ["ask", "fsk"], True, None),
        ("per_signal", ["ask"], False, "Invalid classes for sampling_mode 'per_signal'"),
        ("per_family", ["ook"], False, "Invalid classes for sampling_mode 'per_family'"),
    ],
)
def test_sampling_mode_class_membership_validation(sampling_mode, class_list, should_pass, match_str):
    metadata = DatasetMetadataIn()
    if should_pass:
        config = FullDatasetConfigRequest(
            dataset_id="test_dataset",
            sampling_mode=sampling_mode,
            dataset_metadata=metadata,
            class_list=class_list
        )
        assert config.class_list == class_list
    else:
        with pytest.raises(ValidationError, match=match_str):
            FullDatasetConfigRequest(
                dataset_id="test_dataset",
                sampling_mode=sampling_mode,
                dataset_metadata=metadata,
                class_list=class_list
            )
