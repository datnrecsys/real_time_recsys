from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.domain.model_request import ModelRequest


class BaseDLModel(ABC, nn.Module):
    """
    Abstract base class for a recommender system model that predicts ratings
    """

    @abstractmethod
    def forward(
        self, input_data: ModelRequest
    ):
        ...
    
    def predict(
        self, input_data: ModelRequest
    ):
        output_ratings = self.forward(input_data)
        
        # Apply sigmoid to the output ratings to ensure they are in the range [0, 1]
        output_ratings = torch.sigmoid(output_ratings)
        return output_ratings
    
    
    @abstractmethod
    def recommend(
        self,
        input_data: ModelRequest,
        k: int,
        batch_size: int = 128
    ):
        ...
        
        