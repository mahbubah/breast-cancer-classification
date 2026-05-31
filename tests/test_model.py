"""
Unit tests for src/model.py

What we test:
  - evaluate_model() returns all expected metric keys
  - All metric values are in [0, 1]
  - Predictions are binary (0 or 1)
  - Logistic Regression beats the majority-class baseline (63 %)
  - Decision Tree and Random Forest also beat the baseline
  - ROC-AUC is returned when predict_proba is available
"""
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data, FEATURES, TARGET
from src.preprocessing import split_data, build_pipeline
from src.model import evaluate_model

DATA_PATH = 'data/Breast_cancer_data.csv'

EXPECTED_KEYS = {'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'}
MAJORITY_CLASS_BASELINE = 0.627  # always-Benign accuracy


@pytest.fixture(scope='module')
def data_split():
    df = load_data(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    return split_data(X, y)


def _run(classifier, data_split):
    X_train, X_test, y_train, y_test = data_split
    pipe = build_pipeline(classifier)
    return evaluate_model(classifier.__class__.__name__, pipe, X_train, y_train, X_test, y_test)


# ── Return structure ───────────────────────────────────────────────────────────

def test_evaluate_returns_all_metric_keys(data_split):
    metrics, _, _, _ = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    assert EXPECTED_KEYS == set(metrics.keys())


def test_metrics_are_in_valid_range(data_split):
    metrics, _, _, _ = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    for key in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_predictions_are_binary(data_split):
    _, y_pred, _, _ = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    assert set(y_pred).issubset({0, 1})


def test_roc_auc_is_returned_for_proba_model(data_split):
    metrics, _, y_prob, _ = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    assert y_prob is not None
    assert metrics['ROC-AUC'] is not None


# ── Model performance ─────────────────────────────────────────────────────────

def test_logistic_regression_beats_baseline(data_split):
    metrics, _, _, _ = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    assert metrics['Accuracy'] > MAJORITY_CLASS_BASELINE


def test_decision_tree_beats_baseline(data_split):
    metrics, _, _, _ = _run(DecisionTreeClassifier(random_state=42), data_split)
    assert metrics['Accuracy'] > MAJORITY_CLASS_BASELINE


def test_random_forest_beats_baseline(data_split):
    metrics, _, _, _ = _run(RandomForestClassifier(n_estimators=100, random_state=42), data_split)
    assert metrics['Accuracy'] > MAJORITY_CLASS_BASELINE


def test_both_models_achieve_high_roc_auc(data_split):
    """Both LR and RF should achieve ROC-AUC >= 0.95 — on this dataset LR is competitive."""
    lr_metrics, _, _, _ = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    rf_metrics, _, _, _ = _run(RandomForestClassifier(n_estimators=100, random_state=42), data_split)
    assert lr_metrics['ROC-AUC'] >= 0.95, f"LR ROC-AUC too low: {lr_metrics['ROC-AUC']:.3f}"
    assert rf_metrics['ROC-AUC'] >= 0.95, f"RF ROC-AUC too low: {rf_metrics['ROC-AUC']:.3f}"


def test_fitted_pipeline_is_returned(data_split):
    from sklearn.pipeline import Pipeline
    _, _, _, fitted = _run(LogisticRegression(random_state=42, max_iter=1000), data_split)
    assert isinstance(fitted, Pipeline)
