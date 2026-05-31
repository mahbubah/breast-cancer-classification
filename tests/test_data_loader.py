"""
Unit tests for src/data_loader.py

Beginner-friendly tests that check:
  1. The CSV loads and returns a DataFrame
  2. All required columns are present
  3. There are no missing values
  4. The target column only contains 0 and 1
"""
import pytest
import pandas as pd

from src.data_loader import load_data, validate_data, FEATURES, TARGET

DATA_PATH = 'data/Breast_cancer_data.csv'


@pytest.fixture
def df():
    return load_data(DATA_PATH)


def test_load_data_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


def test_required_columns_are_present(df):
    for col in FEATURES + [TARGET]:
        assert col in df.columns


def test_no_missing_values(df):
    assert df[FEATURES + [TARGET]].isnull().sum().sum() == 0


def test_target_column_is_binary(df):
    assert set(df[TARGET].unique()).issubset({0, 1})

