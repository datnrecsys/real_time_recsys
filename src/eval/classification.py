import os
from typing import Any, Generic, TypeVar

import pandas as pd
from evidently.metrics import (FBetaTopKMetric, NDCGKMetric,
                               PersonalizationMetric, PrecisionTopKMetric,
                               RecallTopKMetric)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

from src.algo.base.base_dl_model import BaseDLModel
from src.domain.model_request import ModelRequest
from src.utils.embedding_id_mapper import IDMapper

T = TypeVar("T", bound=BaseDLModel)

class ItemClassification(Generic[T]):
    def __init__(self, 
                 rec_model: T, 
                 mlf_client: Any | None = None,
                 idm: IDMapper | None = None, 
                 user_col = "user_id", 
                 item_col = "parent_asin",
                 user_idx_col: str = "user_indice",
                 item_idx_col: str = "item_indice",
                 rating_col: str = "rating",
                 timestamp_col: str = "timestamp",
                 batch_size: int = 128):
        
        self.rec_model = rec_model
        self.idm = idm
        self.user_col = user_col
        self.item_col = item_col
        self.user_idx_col = user_idx_col
        self.batch_size = batch_size
        self.item_idx_col = item_idx_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.mlf_client = mlf_client
        

    def create_classification_df(self, input_data: ModelRequest) -> pd.DataFrame:
        
        """
        Create a DataFrame for classification.
        
        Args:
            input_data (ModelRequest): The ModelRequest object containing the input data.
        
        Returns:
            pd.DataFrame: A DataFrame with user and item indices, and predicted ratings.
        """
        df = self.rec_model.predict(input_data)
        df = df.assign(
            user_indice=lambda df: df[self.user_col].apply(lambda x: self.idm.get_user_index(x)),
            item_indice=lambda df: df[self.item_col].apply(lambda x: self.idm.get_item_index(x)),
            rating=lambda df: df[self.rating_col].fillna(0).astype(int),
        )
        return df
        