from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline


def evaluate_model(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    last_step = pipeline.steps[-1][1]
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(last_step, 'predict_proba') else None

    metrics = {
        'Model':     name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1':        f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    return metrics, y_pred, y_prob, pipeline
