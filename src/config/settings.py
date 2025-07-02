from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",        # ← tell Pydantic to drop any env vars you haven’t declared
    )

    # --- Required settings even when working locally. ---

    # PostgreSQL database settings
    POSTGRES_USER: str | None  = None
    POSTGRES_PASSWORD: str | None  = None
    POSTGRES_DB: str | None  = "amazon_rating"
    POSTGRES_HOST: str | None  = None
    POSTGRES_PORT: int | None  = 5432
    POSTGRES_OLTP_SCHEMA: str | None  = "oltp"
    
    #Minio settings
    MINIO_ENDPOINT: str | None  = None
    MINIO_ACCESS_KEY: str | None  = None
    MINIO_SECRET_KEY: str | None  = None
    MINIO_BRONZE_BUCKET: str | None  = None
    
    
settings = Settings()
print(settings)