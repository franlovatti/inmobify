import io
import pickle
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TasadorInmobifyWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper entrenado sobre log1p(precio_m2_usd):
      - .predict(X) -> precio_usd total estimado (re-escalado por superficie_m2)
      - .predict_m2(X) -> precio_m2_usd estimado
    """
    def __init__(self, pipeline_m2):
        self.pipeline_m2 = pipeline_m2

    def fit(self, X, y_usd):
        sup = np.clip(X["superficie_m2"].values, 1.0, None)
        y_log_m2 = np.log1p(y_usd / sup)
        self.pipeline_m2.fit(X, y_log_m2)
        return self

    def predict_m2(self, X):
        pred_log_m2 = self.pipeline_m2.predict(X)
        return np.expm1(pred_log_m2)

    def predict(self, X):
        sup = np.clip(X["superficie_m2"].values, 1.0, None)
        return self.predict_m2(X) * sup


class _InmobifyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "TasadorInmobifyWrapper":
            return TasadorInmobifyWrapper
        return super().find_class(module, name)


def cargar_modelo_seguro(model_bytes: bytes) -> TasadorInmobifyWrapper:
    return _InmobifyUnpickler(io.BytesIO(model_bytes)).load()