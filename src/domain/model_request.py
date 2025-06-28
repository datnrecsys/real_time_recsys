# pip install typeguard
from dataclasses import dataclass
from typing import Optional

import torch
import pandas as pd
from loguru import logger


@dataclass
class ModelRequest:
    user_id: torch.Tensor
    target_item: Optional[torch.Tensor] = None
    recommendation: bool = True

    def __post_init__(self):
        # still keep your cross-field invariant
        if not self.recommendation and self.target_item is None:
            raise ValueError(
                "If recommendation is False, target_item must be provided."
            )
            
            
    @classmethod
    def from_df_for_rec(cls, df: pd.DataFrame, **kwargs) -> 'ModelRequest':
        """
        Convert a DataFrame row to a ModelRequest object.
        
        Args:
            df (pd.DataFrame): A DataFrame row containing user_id and target_item.
        
        Returns:
            ModelRequest: An instance of ModelRequest with user_id and target_item.
        """
        user_col = kwargs.get('user_col', 'user_indice')
        
        device = kwargs.get('device', 'cpu')
        
        logger.info(f"Use user_col={user_col}")
        
        user_id = torch.tensor(df[user_col].values, device=device)
        
        return cls(user_id=user_id)

@dataclass(kw_only=True)
class SequenceModelRequest(ModelRequest):
    item_sequence: torch.Tensor

    @classmethod
    def from_df_for_rec(cls, df: pd.DataFrame, **kwargs) -> 'SequenceModelRequest':
        user_col = kwargs.get('user_col', 'user_indice')
        item_seq = kwargs.get('item_sequence_col', 'item_sequence')
        device = kwargs.get('device', 'cpu')
        
        logger.info(f"Use user_col={user_col}")
        logger.info(f"Use item_sequence_col={item_seq}")

        user_id = torch.tensor(df[user_col].values, device=device)
        item_sequence = torch.tensor(df[item_seq].values.tolist(), device=device)

        return cls(user_id=user_id, item_sequence=item_sequence)
    
    
