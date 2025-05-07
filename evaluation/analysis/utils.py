from pathlib import Path
from typing import Union

import pandas as pd


def load_report_files(
    directory: Union[str, Path], suffix: str = "_report.csv"
) -> pd.DataFrame:
    """
    Load all evaluation report CSV files from a directory into a single DataFrame.

    Args:
        directory (Union[str, Path]): Path to the directory containing report files.
        suffix (str): Filename suffix to identify report files. Defaults to "_report.csv".

    Returns:
        pd.DataFrame: Concatenated DataFrame with all report data and model identifiers.

    Raises:
        FileNotFoundError: If no matching report files are found in the directory.
    """
    directory = Path(directory)
    all_files = list(directory.glob(f"*{suffix}"))
    frames = []

    for file in all_files:
        df = pd.read_csv(file)
        df["model_id"] = file.stem.replace(suffix, "")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV reports found in {directory}")

    return pd.concat(frames, ignore_index=True)
