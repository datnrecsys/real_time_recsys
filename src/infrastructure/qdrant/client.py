from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config.infra import InfraSettings
from src.infrastructure.base.singleton import SingletonMeta


class QdrantClientConnection(metaclass=SingletonMeta):
    """
    Singleton class to manage the Qdrant client connection.
    """
    def __init__(self, settings: InfraSettings = InfraSettings()):
        if not settings.qdrant_host:
            raise ValueError("Qdrant host is not set in the configuration.")
        
        try:
            self.client = QdrantClient(
                url=settings.qdrant_host,
                port=settings.qdrant_port,
                prefer_grpc=False
            )
            
            uri = f"{settings.qdrant_host}:{settings.qdrant_port}"
            logger.info(f"Connected to Qdrant at {uri}.")
            
        except UnexpectedResponse:
            logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
            raise
        
        
def get_qdrant_client() -> QdrantClient:
    """
    Get the singleton instance of the Qdrant client.
    
    Returns:
        QdrantClient: The singleton instance of the Qdrant client.
    """
    return QdrantClientConnection().client  # This will ensure only one instance is created.