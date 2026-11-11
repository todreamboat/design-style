"""
Project configuration management
Configuration validation and management using Pydantic
"""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import AliasChoices, Field, validator
except ImportError:
    # If pydantic_settings is not available, use basic configuration
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(**kwargs):
        def wrapper(default):
            return default
        return wrapper
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class Settings(BaseSettings):
    """Project settings class"""
    
    # Project basic settings
    project_name: str = "Meta Designer Training System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Path configuration
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    log_dir: Path = Field(default_factory=lambda: Path("logs"))
    temp_dir: Path = Field(default_factory=lambda: Path("data/temp"))
    
    # API configuration
    api_vlm_api_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    # Secrets must be injected through environment variables. Never commit them.
    api_vlm_api_key: str = Field(default="", validation_alias=AliasChoices("DST_API_VLM_API_KEY", "OPENROUTER_API_KEY", "api_vlm_api_key"))
    api_vlm_model_name: str = "qwen/qwen3.8-max-0902"
    api_vlm_max_tokens: int = 500
    api_vlm_temperature: float = 0.2
    api_vlm_timeout: int = 60
    
    api_llm_api_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    api_llm_api_key: str = Field(default="", validation_alias=AliasChoices("DST_API_LLM_API_KEY", "OPENROUTER_API_KEY", "api_llm_api_key"))
    api_llm_model_name: str = "qwen/qwen3.8-max-0902"
    api_llm_max_tokens: int = 800
    api_llm_temperature: float = 0.3
    api_llm_timeout: int = 90
    
    api_max_retries: int = 3
    api_retry_delay: float = 1.0
    api_rate_limit_per_minute: int = 60
    
    # Storage configuration
    designer_styles_dir: str = "data/designer_styles"
    max_images_per_designer: int = 30
    supported_image_formats: list = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    max_image_size_mb: int = 10
    
    # Hash configuration
    hash_algorithm: str = "sha256"
    hash_length: int = 12
    
    # File naming configuration
    image_name_format: str = "{index:02d}.png"
    mapping_file_name: str = "designer_mapping.json"
    metadata_file_name: str = "metadata.json"
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    log_rotation: str = "1 week"
    log_retention: str = "1 month"
    
    # Security configuration
    enable_file_validation: bool = True
    max_concurrent_operations: int = 10
    
    @validator('data_dir', 'log_dir', 'temp_dir', always=True)
    def ensure_absolute_path(cls, v, values):
        """Ensure paths are absolute"""
        if not v.is_absolute():
            base_dir = values.get('base_dir', Path(__file__).parent.parent)
            return base_dir / v
        return v
    
    @validator('designer_styles_dir', always=True)
    def ensure_designer_styles_absolute(cls, v, values):
        """Ensure designer styles directory is absolute path"""
        path = Path(v)
        if not path.is_absolute():
            base_dir = values.get('base_dir', Path(__file__).parent.parent)
            return str(base_dir / v)
        return v
    
    def create_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.data_dir,
            self.log_dir,
            self.temp_dir,
            Path(self.designer_styles_dir)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = (".env", str(Path(__file__).resolve().parent.parent / "runtime" / "openrouter.env"))
        env_prefix = "DST_"  # Designer Style Training prefix


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings"""
    global _settings
    _settings = None
    return get_settings()
