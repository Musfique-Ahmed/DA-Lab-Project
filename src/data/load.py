"""Loader for application_train.csv with a loud shape assertion.

Per the master prompt, the only dataset file in scope for this project is
`.home-credit-default-risk/application_train.csv`. The loader enforces the
expected shape (307,511 rows, 122 columns) so any future corruption or
mismatched file is caught immediately.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical location of the only in-scope dataset.
DEFAULT_DATA_PATH = Path(".home-credit-default-risk") / "application_train.csv"

# Shape contract for application_train.csv. If Home Credit ever refreshes
# the dataset, this assertion is the first thing that should fail.
EXPECTED_SHAPE: tuple[int, int] = (307_511, 122)


def load_application_train(path: str | Path | None = None) -> pd.DataFrame:
    """Load application_train.csv and assert it matches the expected shape.

    Parameters
    ----------
    path : str | Path | None
        Path to application_train.csv. Defaults to
        ``.home-credit-default-risk/application_train.csv``.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified application_train dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the resolved path.
    ValueError
        If the loaded dataframe's shape does not match
        ``EXPECTED_SHAPE = (307511, 122)``.
    """
    resolved = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"application_train.csv not found at {resolved}. "
            "Place the dataset at this path or pass `path=...` explicitly."
        )

    df = pd.read_csv(resolved)

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"application_train.csv has unexpected shape {df.shape}; "
            f"expected {EXPECTED_SHAPE}. Did the dataset change upstream?"
        )

    return df


if __name__ == "__main__":
    # Quick CLI smoke test.
    df = load_application_train()
    print(f"Loaded application_train.csv with shape {df.shape}")
    print(f"Columns: {df.shape[1]}, Target positive rate: {df['TARGET'].mean():.4%}")
