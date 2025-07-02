from abc import ABC, abstractmethod
import pandas as pd
import mlflow

from src.algo.base.base_dl_model import BaseDLModel
from src.domain.model_request import ModelRequest

class BasePipeline(ABC):
    def __init__(self, model: BaseDLModel, experiment_name: str):
        self.model = model
        mlflow.set_experiment(experiment_name)
        self._init_evidently()

    @abstractmethod
    def _init_evidently(self):
        """Create self.profile and self.dashboard for this pipeline"""

    @abstractmethod
    def _build_requests(self, df: pd.DataFrame) -> ModelRequest:
        """Turn raw DataFrame rows into ModelRequest objects"""

    @abstractmethod
    def _call_model(self, reqs: ModelRequest) -> list:
        """Call model.predict or model.recommend"""

    @abstractmethod
    def _assemble_df(self, df: pd.DataFrame, outputs: list) -> pd.DataFrame:
        """Build a DataFrame of results (predictions or recs)"""

    @abstractmethod
    def _log_artifacts(self, result_df: pd.DataFrame):
        """Log CSV + Evidently artifacts + any custom metrics to MLflow"""

    def run(self, input_df: pd.DataFrame) -> pd.DataFrame:
        reqs = self._build_requests(input_df)
        with mlflow.start_run():
            outputs = self._call_model(reqs)
            result_df = self._assemble_df(input_df, outputs)
            self._log_artifacts(result_df)
        return result_df
