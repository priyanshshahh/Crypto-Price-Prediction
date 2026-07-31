"""Walk-forward splits with purge + embargo for time-series CV.

Used by multi-horizon price models and paper evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_end: int   # exclusive
    test_start: int  # inclusive
    test_end: int    # exclusive


def expanding_walk_forward(
    n: int,
    *,
    min_train: int = 180,
    test_size: int = 30,
    embargo: int = 5,
    purge: int = 0,
    step: int | None = None,
) -> list[Fold]:
    """Expanding-window walk-forward with optional purge/embargo.

    purge: drop this many samples at the end of train (feature lookback / label horizon).
    embargo: gap between train_end and test_start.
    """
    if n < min_train + test_size:
        return []
    step = step or test_size
    folds: list[Fold] = []
    test_start = min_train + embargo
    while test_start + test_size <= n:
        train_end = test_start - embargo - purge
        if train_end < min_train:
            test_start += step
            continue
        folds.append(Fold(train_end=train_end, test_start=test_start, test_end=test_start + test_size))
        test_start += step
    return folds


def chronological_holdout(n: int, test_frac: float = 0.2) -> Fold:
    split = int(n * (1 - test_frac))
    return Fold(train_end=split, test_start=split, test_end=n)
