"""
Unit tests for src/data_loader.py

What we test:
  - CSV loads and has the right shape
  - All required columns are present
  - No missing values
  - Target column contains only 0 and 1
  - validate_data() raises clear errors on bad input
"""
import pytest
import pandas as pd

from src.data_loader import load_data, validate_data, FEATURES, TARGET

DATA_PATH = 'data/Breast_cancer_data.csv'


@pytest.fixture(scope='module')
def df():
    return load_data(DATA_PATH)


# ── Shape & content ────────────────────────────────────────────────────────────

def test_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


def test_row_count(df):
    assert df.shape[0] == 569


def test_column_count(df):
    assert df.shape[1] == 6


def test_feature_columns_present(df):
    for col in FEATURES:
        assert col in df.columns, f"Missing feature column: {col}"


def test_target_column_present(df):
    assert TARGET in df.columns


def test_no_missing_values(df):
    assert df[FEATURES + [TARGET]].isnull().sum().sum() == 0


def test_target_is_binary(df):
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_class_counts(df):
    counts = df[TARGET].value_counts()
    assert counts[0] == 212, "Expected 212 Malignant cases"
    assert counts[1] == 357, "Expected 357 Benign cases"


# ── validate_data() ────────────────────────────────────────────────────────────

def test_validate_passes_on_good_data(df):
    assert validate_data(df) is True


def test_validate_raises_on_missing_feature_column(df):
    broken = df.drop(columns=['mean_radius'])
    with pytest.raises(ValueError, match="Feature column 'mean_radius'"):
        validate_data(broken)


def test_validate_raises_on_missing_target_column(df):
    broken = df.drop(columns=[TARGET])
    with pytest.raises(ValueError, match="Target column 'diagnosis'"):
        validate_data(broken)


def test_validate_raises_on_missing_values(df):
    broken = df.copy()
    broken.loc[0, 'mean_radius'] = None
    with pytest.raises(ValueError, match="Missing values"):
        validate_data(broken)


def test_validate_raises_on_unexpected_target_values(df):
    broken = df.copy()
    broken.loc[0, TARGET] = 99
    with pytest.raises(ValueError, match="unexpected values"):
        validate_data(broken)
