import numpy as np


class EnsembleModel:
    """Soft-voting ensemble: averages predict_proba across multiple classifiers."""

    def __init__(self, models: list, avg_importances=None):
        self.models = models
        self._avg_importances = avg_importances

    @property
    def feature_importances_(self):
        if self._avg_importances is not None:
            return self._avg_importances
        imps = [m.feature_importances_ for m in self.models if hasattr(m, "feature_importances_")]
        return np.mean(imps, axis=0) if imps else None

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
