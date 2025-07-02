from pydantic_settings import BaseSettings, SettingsConfigDict


class InfraSettings(BaseSettings):
    """
    Configuration settings for the infrastructure.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",        # ← tell Pydantic to drop any env vars you haven’t declared
    )
    
    # Qdrant settings:
    qdrant_host: str | None = None
    qdrant_port: int | None = 6333
    qdrant_collection_name: str | None = "item2vec"
    
    