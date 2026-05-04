from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    MONGODB_URI: str = "mongodb://localhost:27017/travela"
    PORT: int = 8000
    DEBUG: bool = False

    # Database name (extracted from URI or specified)
    DB_NAME: str = "travela"

    # Cache settings
    CACHE_TTL_HOMEPAGE: int = 120  # 2 minutes
    CACHE_TTL_SIMILAR: int = 600   # 10 minutes
    CACHE_TTL_POPULARITY: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
