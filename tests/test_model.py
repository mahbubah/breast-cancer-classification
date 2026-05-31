"""
Unit tests for src/model.py

Beginner-friendly tests that check:
  1. evaluate_model returns all the expected metric keys
  2. Every metric value is a number between 0 and 1
  3. Predictions are binary — only 0 or 1
"""
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_data, FEATURES, TARGET
from src.preprocessing import split_data, build_pipeline
from src.model import evaluate_model

DATA_PATH = 'data/Breast_cancer_data.csv'


@pytest.fixture
def trained_result():
    df = load_data(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_pipeline(LogisticRegression(random_state=42, max_iter=1000))
    return evaluate_model('Logistic Regression', pipeline, X_train, y_train, X_test, y_test)


def test_evaluate_model_returns_all_metric_keys(trained_result):
    metrics, _, _, _ = trained_result
    for key in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']:
        assert key in metrics


def test_all_metric_values_are_between_0_and_1(trained_result):
    metrics, _, _, _ = trained_result
    for key in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']:
        assert 0.0 <= metrics[key] <= 1.0


def test_predictions_are_binary(trained_result):
    _, y_pred, _, _ = trained_result
    assert set(y_pred).issubset({0, 1})
