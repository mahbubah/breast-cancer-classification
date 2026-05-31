import pandas as pd

FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
TARGET = 'diagnosis'
LABEL_MAP = {0: 'Malignant', 1: 'Benign'}


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def validate_data(df: pd.DataFrame) -> bool:
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            kind = 'Feature' if col != TARGET else 'Target'
            raise ValueError(f"{kind} column '{col}' not found in dataframe")
    missing = df[FEATURES + [TARGET]].isnull().sum()
    if missing.any():
        raise ValueError(f"Missing values found:\n{missing[missing > 0]}")
    unexpected = set(df[TARGET].unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"Target contains unexpected values: {unexpected}")
    return True
