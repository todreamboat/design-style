"""
API configuration management
Manage configuration information for various APIs
"""

from typing import Optional, Dict, Any

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
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


class APIConfig(BaseSettings):
    """API configuration class"""
    
    # VLM (Vision Language Model) API configuration
    vlm_api_endpoint: Optional[str] = None
    vlm_api_key: Optional[str] = None
    vlm_model_name: str = "gpt-4o"
    vlm_max_tokens: int = 500
    vlm_temperature: float = 0.2
    vlm_timeout: int = 60
    
    # LLM (Large Language Model) API configuration
    llm_api_endpoint: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model_name: str = "gpt-4o"
    llm_max_tokens: int = 800
    llm_temperature: float = 0.3
    llm_timeout: int = 90
    
    # General API configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 60
    
    @validator('vlm_api_endpoint', 'llm_api_endpoint')
    def validate_api_endpoint(cls, v):
        """Validate API endpoint format"""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('API endpoint must start with http:// or https://')
        return v
    
    @validator('vlm_api_key', 'llm_api_key')
    def validate_api_key(cls, v):
        """Validate API key format"""
        if v and len(v) < 10:
            raise ValueError('API key seems too short')
        return v
    
    def get_vlm_config(self) -> Dict[str, Any]:
        """Get VLM configuration dictionary"""
        return {
            "api_endpoint": self.vlm_api_endpoint,
            "api_key": self.vlm_api_key,
            "model_name": self.vlm_model_name,
            "max_tokens": self.vlm_max_tokens,
            "temperature": self.vlm_temperature,
            "timeout": self.vlm_timeout
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary"""
        return {
            "api_endpoint": self.llm_api_endpoint,
            "api_key": self.llm_api_key,
            "model_name": self.llm_model_name,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
            "timeout": self.llm_timeout
        }
    
    def is_vlm_configured(self) -> bool:
        """Check if VLM is configured"""
        return self.vlm_api_endpoint is not None and self.vlm_api_key is not None
    
    def is_llm_configured(self) -> bool:
        """Check if LLM is configured"""
        return self.llm_api_endpoint is not None and self.llm_api_key is not None
    
    class Config:
        env_file = ".env"
        env_prefix = "API_"


# 全局API配置实例
_api_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """获取全局API配置实例"""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config
