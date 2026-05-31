from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_pipeline(classifier: ClassifierMixin) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', classifier),
    ])
