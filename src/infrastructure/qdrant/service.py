from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import Batch, Filter, PointStruct, VectorParams


class QdrantService:
    """
    Service class to manage interactions with the Qdrant client.
    """
    
    def __init__(self, client: QdrantClient):
        self.client = client
        
    def get_collection(self, collection_name: str):
        """
        Retrieve a collection by its name.
        
        Args:
            collection_name (str): The name of the collection to retrieve.
        
        Returns:
            Collection: The requested collection.
        """
        return self.client.get_collection(collection_name)

    def create_vector_collection(self, 
                                 collection_name: str, 
                                 vector_size: int, 
                                 distance: Literal["Cosine", "Euclidean", "Dot", "Manhattan"] = "Cosine",
                                 **kwargs
                                 ) -> None:
        """
        Create a new vector collection.
        
        Args:
            collection_name (str): The name of the collection to create.
            vector_size (int): The size of the vectors in the collection.
            distance (str): The distance metric to use for the collection.
        
        Returns:
            Collection: The created collection.
        """
        return self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size = vector_size, distance = distance),
            
        )
        
    def recreate_vector_collection(self, 
                                   collection_name: str, 
                                   vector_size: int, 
                                   distance: Literal["Cosine", "Euclidean", "Dot", "Manhattan"] = "Cosine",
                                   **kwargs) -> None:
        """
        Recreate a vector collection by deleting and creating it again.
        
        Args:
            collection_name (str): The name of the collection to recreate.
            vector_size (int): The size of the vectors in the collection.
            distance (str): The distance metric to use for the collection.
        """
        return self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size = vector_size, distance = distance),
            **kwargs
        )
        
    def upsert_points(self, 
                      collection_name: str, 
                      points: list[PointStruct] | Batch, 
                      **kwargs) -> None:
        """        
        Upsert points into a collection.
        Args:
            collection_name (str): The name of the collection to upsert points into.
            points (list[PointStruct] | Batch): The points to upsert.
        """
        return self.client.upsert(
            collection_name=collection_name,
            points=points,
            **kwargs
        )
        
    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: Filter | None = None,
        limit: int = 3,
    ) -> list:
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        
    def scroll(self, collection_name: str, limit: int):
        return self.client.scroll(collection_name=collection_name, limit=limit)
