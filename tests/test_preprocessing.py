"""
Unit tests for src/preprocessing.py

Beginner-friendly tests that check:
  1. split_data produces train and test sets that add up to the full dataset
  2. split_data keeps the right proportion of test rows (~20%)
  3. build_pipeline returns a Pipeline with a scaler and a classifier
"""
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import load_data, FEATURES, TARGET
from src.preprocessing import split_data, build_pipeline

DATA_PATH = 'data/Breast_cancer_data.csv'


@pytest.fixture
def X_y():
    df = load_data(DATA_PATH)
    return df[FEATURES], df[TARGET]


def test_split_sizes_add_up(X_y):
    X, y = X_y
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)


def test_split_test_is_20_percent(X_y):
    X, y = X_y
    _, X_test, _, _ = split_data(X, y)
    assert abs(len(X_test) / len(X) - 0.2) < 0.02


def test_build_pipeline_returns_pipeline():
    pipeline = build_pipeline(LogisticRegression())
    assert isinstance(pipeline, Pipeline)


def test_pipeline_has_scaler_and_classifier():
    pipeline = build_pipeline(LogisticRegression())
    assert 'scaler' in pipeline.named_steps
    assert 'clf' in pipeline.named_steps
