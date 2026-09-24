import numpy as np

from src.train import (
    calculate_scale_pos_weight,
)


def test_scale_pos_weight_handles_imbalance():
    y = np.array(
        [0] * 90 + [1] * 10
    )

    result = (
        calculate_scale_pos_weight(y)
    )

    assert result == 9.0


def test_scale_pos_weight_rejects_no_positive():
    y = np.array(
        [0, 0, 0, 0]
    )

    try:
        calculate_scale_pos_weight(y)
        assert False
    except ValueError as exc:
        assert (
            "No positive samples"
            in str(exc)
        )