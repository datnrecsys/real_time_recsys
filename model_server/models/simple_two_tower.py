from pydantic import BaseModel
from typing import List

class SimpleTwoTowerModelInput(BaseModel):
    user_id: List[str]
    item_id: List[str]
