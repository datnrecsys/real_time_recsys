from pydantic import BaseModel


class AmazonDatasetConfig(BaseModel):
    
    dataset_name: str = "amazon_rating_review"
    
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"
    
    # User and Item ID columns
    user_col: str = "user_id"
    item_col: str = "parent_asin"
    
    # User and Item indice columns
    