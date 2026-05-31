"""
Unit tests for src/preprocessing.py

What we test:
  - Train/test split produces the right sizes
  - Stratification keeps the class ratio consistent
  - build_pipeline() produces a correct Pipeline object
  - The pipeline fits and predicts without error
  - Predictions are all 0 or 1 (binary)
"""
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data, FEATURES, TARGET
from src.preprocessing import split_data, build_pipeline

DATA_PATH = 'data/Breast_cancer_data.csv'


@pytest.fixture(scope='module')
def X_y():
    df = load_data(DATA_PATH)
    return df[FEATURES], df[TARGET]


@pytest.fixture(scope='module')
def split(X_y):
    X, y = X_y
    return split_data(X, y)


# ── split_data() ───────────────────────────────────────────────────────────────

def test_split_total_rows(X_y, split):
    X, _ = X_y
    X_train, X_test, _, _ = split
    assert len(X_train) + len(X_test) == len(X)


def test_split_test_size_is_20_percent(X_y, split):
    X, _ = X_y
    _, X_test, _, _ = split
    assert abs(len(X_test) / len(X) - 0.2) < 0.02


def test_split_train_has_more_rows(split):
    X_train, X_test, _, _ = split
    assert len(X_train) > len(X_test)


def test_split_stratification(split):
    """Malignant proportion should be similar in train and test."""
    _, _, y_train, y_test = split
    train_ratio = (y_train == 0).sum() / len(y_train)
    test_ratio  = (y_test == 0).sum()  / len(y_test)
    assert abs(train_ratio - test_ratio) < 0.02


def test_split_no_overlap(split):
    X_train, X_test, _, _ = split
    train_idx = set(X_train.index)
    test_idx  = set(X_test.index)
    assert train_idx.isdisjoint(test_idx)


# ── build_pipeline() ───────────────────────────────────────────────────────────

def test_build_pipeline_returns_pipeline():
    pipe = build_pipeline(LogisticRegression())
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_scaler_step():
    pipe = build_pipeline(LogisticRegression())
    assert 'scaler' in pipe.named_steps
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)


def test_pipeline_has_clf_step():
    clf = LogisticRegression()
    pipe = build_pipeline(clf)
    assert 'clf' in pipe.named_steps
    assert pipe.named_steps['clf'] is clf


def test_pipeline_fits_and_predicts(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_pipeline(LogisticRegression(random_state=42, max_iter=1000))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)


def test_pipeline_predictions_are_binary(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline(LogisticRegression(random_state=42, max_iter=1000))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds).issubset({0, 1})
