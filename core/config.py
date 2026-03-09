"""Core configuration and global settings for QuantumLab."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration for the application."""
    
    # Randomness
    DEFAULT_SEED: int = 42
    
    # Physics defaults
    DEFAULT_GRID_SIZE: int = 80
    DEFAULT_MC_BATCH_SIZE: int = 200_000
    
    # API settings
    API_TITLE: str = "QuantumLab API"
    API_VERSION: str = "2.0.0"
    
    class Config:
        env_prefix = "QUANTUM_"


settings = Settings()
