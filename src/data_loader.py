from __future__ import annotations

import io
from typing import Tuple, Optional

import pandas as pd


def _read_with_sep(text: str, sep: str) -> pd.DataFrame:
    """
    Read text into a DataFrame using a given separator.
    sep can be ',' ';' '\t' or r'\s+' (whitespace regex).
    """
    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",          # allows regex separators like r'\s+'
        comment="#",              # ignore comment lines starting with '#'
        skip_blank_lines=True,
    )


def load_measurement(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Streamlit uploaded_file -> (df, detected_separator_label)

    Tries common separators and returns the one that produces the "best" table.
    """
    # Read bytes -> text safely
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="replace")

    # Try separators from most common to least
    candidates = [
        (",", "comma (,)"),
        (";", "semicolon (;)"),
        ("\t", "tab (\\t)"),
        (r"\s+", "whitespace"),
    ]

    best_df: Optional[pd.DataFrame] = None
    best_label = "unknown"
    best_score = -1

    for sep, label in candidates:
        try:
            df = _read_with_sep(text, sep)

            # Basic cleanup: drop completely empty columns
            df = df.dropna(axis=1, how="all")

            # Score: prefer more columns + more rows (but require at least 2 columns)
            nrows, ncols = df.shape
            if ncols < 2 or nrows < 1:
                continue

            score = (ncols * 10) + nrows  # weight columns a bit more
            if score > best_score:
                best_score = score
                best_df = df
                best_label = label
        except Exception:
            continue

    if best_df is None:
        # Last resort: try letting pandas infer (works for many CSVs)
        best_df = pd.read_csv(io.StringIO(text), engine="python", comment="#")
        best_df = best_df.dropna(axis=1, how="all")
        best_label = "pandas inferred"

    return best_df, best_label
