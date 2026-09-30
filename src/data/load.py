"""Loader for application_train.csv with a loud shape assertion.

Per the master prompt, the only dataset file in scope for this project is
``.home-credit-default-risk/application_train.csv`` (or, for the MLSD
pipeline, ``data/raw/application_train.csv``). The loader enforces the
expected shape (307,510 rows, 122 columns) so any future corruption or
mismatched file is caught immediately.

The shape check can be skipped by setting the environment variable
``CREDIT_RISK_SKIP_SHAPE_CHECK=1`` — useful only for synthetic validation
data (see ``scripts/_make_synthetic_for_validation.py``). Production
runs always have the check enabled.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Canonical location of the only in-scope dataset. The MLSD pipeline
# uses data/raw/; the DA dashboard used .home-credit-default-risk/.
# We default to data/raw/ for the MLSD project; the DA dashboard
# imports load_application_train with an explicit `path=`.
DEFAULT_DATA_PATH = Path("data") / "raw" / "application_train.csv"

# Shape contract for application_train.csv. If Home Credit ever refreshes
# the dataset, this assertion is the first thing that should fail.
EXPECTED_SHAPE: tuple[int, int] = (307_510, 122)


def load_application_train(path: str | Path | None = None) -> pd.DataFrame:
    """Load application_train.csv and assert it matches the expected shape.

    Parameters
    ----------
    path : str | Path | None
        Path to application_train.csv. Defaults to
        ``data/raw/application_train.csv``.

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
        ``EXPECTED_SHAPE = (307510, 122)`` and the shape check is not
        explicitly disabled via ``CREDIT_RISK_SKIP_SHAPE_CHECK=1``.
    """
    resolved = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"application_train.csv not found at {resolved}. "
            "Place the dataset at this path or pass `path=...` explicitly."
        )

    df = pd.read_csv(resolved)

    skip_check = os.environ.get("CREDIT_RISK_SKIP_SHAPE_CHECK") == "1"
    if not skip_check and df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"application_train.csv has unexpected shape {df.shape}; "
            f"expected {EXPECTED_SHAPE}. Did the dataset change upstream? "
            f"(Set CREDIT_RISK_SKIP_SHAPE_CHECK=1 to override — synthetic test data only.)"
        )

    return df


if __name__ == "__main__":
    # Quick CLI smoke test.
    df = load_application_train()
    print(f"Loaded application_train.csv with shape {df.shape}")
    print(f"Columns: {df.shape[1]}, Target positive rate: {df['TARGET'].mean():.4%}")
