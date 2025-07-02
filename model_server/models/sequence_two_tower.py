from typing import List, Optional

from pydantic import BaseModel


class SequenceTwoTowerModelInput(BaseModel):
    item_sequences: List[List[str]]  # List of item sequences, each sequence is a list of item IDs
    item_ids: Optional[List[str]] = None  # Optional: List of item IDs for which to predict scores
