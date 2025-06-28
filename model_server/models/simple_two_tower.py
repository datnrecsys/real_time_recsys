from typing import List

from pydantic import BaseModel


class SimpleTwoTowerModelInput(BaseModel):
    user_id: List[str]
    item_id: List[str]
