"""
API client module
Handle interactions with LLM and VLM APIs
"""

import base64
import requests
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger
from PIL import Image
import io


class APIClient:
    """API client class"""
    
    def __init__(self, api_endpoint: str, api_key: str, model_name: str, 
                 max_tokens: int = 1000, temperature: float = 0.3, 
                 timeout: int = 60, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _make_request(self, payload: Dict[str, Any]) -> Optional[str]:
        """Send API request"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"API response for {self.model_name}: {result}")
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        if content and content.strip():
                            logger.info(f"API call successful: {self.model_name}")
                            return content
                        else:
                            logger.error(f"API response has empty content for {self.model_name}: {result}")
                    else:
                        logger.error(f"API response format error for {self.model_name}: {result}")
                        
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API request rate limited for {self.model_name}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"API request failed for {self.model_name}: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API request exception for {self.model_name} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
        return None
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image to base64"""
        try:
            logger.debug(f"Encoding image: {image_path}")
            
            # Open and resize image to reduce token consumption
            with Image.open(image_path) as img:
                # Log original image info
                logger.debug(f"Original image: {img.size}, mode: {img.mode}, format: {img.format}")
                
                # Maintain aspect ratio, maximum size 512x512
                # Some models work better with smaller images
                max_size = 512
                if "gemini" in self.model_name.lower():
                    max_size = 768  # Gemini may handle larger images better
                
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image: {img.size}")
                
                # Convert to RGB mode (if needed)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    logger.debug(f"Converted to RGB mode")
                
                # Save to byte buffer
                buffer = io.BytesIO()
                # Use higher quality for better analysis
                quality = 90 if "gemini" in self.model_name.lower() else 85
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                
                # Encode to base64
                img_bytes = buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                logger.debug(f"Base64 encoded length: {len(img_base64)} characters")
                return img_base64
                
        except Exception as e:
            logger.error(f"Image encoding failed {image_path}: {str(e)}")
            return None
    
    def analyze_image_vlm(self, image_path: str, analysis_prompt: str) -> Optional[str]:
        """Analyze image using VLM"""
        img_base64 = self._encode_image_to_base64(image_path)
        if not img_base64:
            return None
        
        # Different models may have different payload formats
        if "gemini" in self.model_name.lower():
            # Gemini models may prefer a different format
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "high"  # Add detail parameter for better analysis
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        else:
            # Standard format for other models
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        
        logger.debug(f"Sending VLM request with model: {self.model_name}")
        return self._make_request(payload)
    
    def generate_text_llm(self, prompt: str) -> Optional[str]:
        """Generate text using LLM"""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        return self._make_request(payload)


def create_vlm_client(api_endpoint: str, api_key: str, model_name: str = "gpt-4o") -> APIClient:
    """Create VLM client with model-specific configurations"""
    
    # Different models may need different token limits
    if "gemini" in model_name.lower():
        max_tokens = 2000  # Gemini needs more tokens for vision tasks
        temperature = 0.1  # Lower temperature for more consistent results
        timeout = 120  # Longer timeout for complex reasoning
    elif "doubao" in model_name.lower():
        max_tokens = 2000  # Doubao with thinking needs more tokens
        temperature = 0.2
        timeout = 90
    elif "qwen" in model_name.lower():
        max_tokens = 2000  # Qwen is efficient
        temperature = 0.2
        timeout = 60
    else:  # gpt-4o and others
        max_tokens = 2000
        temperature = 0.2
        timeout = 60
    
    return APIClient(
        api_endpoint=api_endpoint,
        api_key=api_key,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        max_retries=3,
        retry_delay=1.0
    )


def create_llm_client(api_endpoint: str, api_key: str, model_name: str = "gpt-4o") -> APIClient:
    """Create LLM client with model-specific configurations"""
    
    # Different models may need different token limits
    if "gemini" in model_name.lower():
        max_tokens = 3000  # Gemini needs more tokens for synthesis
        temperature = 0.2
        timeout = 150
    elif "doubao" in model_name.lower():
        max_tokens = 2000  # Doubao with thinking needs more tokens
        temperature = 0.3
        timeout = 120
    elif "qwen" in model_name.lower():
        max_tokens = 1500  # Qwen is efficient
        temperature = 0.3
        timeout = 90
    else:  # gpt-4o and others
        max_tokens = 1200
        temperature = 0.3
        timeout = 90
    
    return APIClient(
        api_endpoint=api_endpoint,
        api_key=api_key,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        max_retries=3,
        retry_delay=1.0
    )
