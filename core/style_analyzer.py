"""
Designer style analysis module
Handle style extraction and analysis of designer images
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
import time

from core.api_client import create_vlm_client, create_llm_client
from core.style_schema import (FIELDS, SCHEMA_VERSION, AGGREGATION_VERSION, OBSERVE_PROMPT,
                               synthesis_prompt, validate_profile, validate_synthesis)


class StyleAnalyzer:
    """Designer style analyzer"""
    
    def __init__(self, vlm_config: Dict[str, Any], llm_config: Dict[str, Any]):
        self.vlm_client = create_vlm_client(
            api_endpoint=vlm_config["api_endpoint"],
            api_key=vlm_config["api_key"],
            model_name=vlm_config["model_name"]
        )
        
        self.llm_client = create_llm_client(
            api_endpoint=llm_config["api_endpoint"],
            api_key=llm_config["api_key"],
            model_name=llm_config["model_name"]
        )
        self.llm_client.max_tokens = llm_config.get("max_tokens", 8192)
        
        self.style_dimensions = {field: [] for field in FIELDS}

    def get_vlm_analysis_prompt(self) -> str:
        return OBSERVE_PROMPT

    def get_llm_synthesis_prompt(self, individual_analyses: List[str]) -> str:
        return synthesis_prompt(individual_analyses)

    def analyze_single_image(self, image_path: str) -> Optional[str]:
        """Analyze single image"""
        logger.info(f"Starting image analysis: {image_path}")
        
        prompt = self.get_vlm_analysis_prompt()
        result = self.vlm_client.analyze_image_vlm(image_path, prompt)
        
        if result:
            logger.info(f"Image analysis completed: {image_path}")
            return result
        else:
            logger.error(f"Image analysis failed: {image_path}")
            return None
    
    def synthesize_style(self, individual_analyses: List[str]) -> Optional[Dict[str, Any]]:
        """Semantic distillation, summarize designer style"""
        logger.info(f"Starting semantic distillation with {len(individual_analyses)} analysis results")
        
        try:
            prompt = self.get_llm_synthesis_prompt(individual_analyses)
        except (ValueError, TypeError):
            logger.error("Invalid source observations; aggregation stopped")
            return None
        result = self.llm_client.generate_text_llm(prompt)
        
        if result:
            try:
                # Try to parse JSON
                if result.strip().startswith('{'):
                    style_data = json.loads(result)
                else:
                    # If not JSON format, try to extract JSON part
                    import re
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        style_data = json.loads(json_match.group())
                    else:
                        logger.error("No JSON profile; stopping without invented confidence")
                        return None
                
                style_data.pop("confidence_score", None)
                validate_synthesis(style_data, individual_analyses)
                logger.info("Semantic distillation completed")
                return style_data
                
            except (ValueError, TypeError):
                logger.error("Semantic distillation result JSON parsing failed")
                return None
        else:
            logger.error("Semantic distillation failed")
            return None
    
    def analyze_designer_style(self, designer_folder: str, designer_name: str, designer_hash: str, progress_callback=None) -> Optional[Dict[str, Any]]:
        """Analyze designer style"""
        logger.info(f"Starting designer style analysis: {designer_name} ({designer_hash})")
        
        # Get all images in the folder
        folder_path = Path(designer_folder)
        if not folder_path.exists():
            logger.error(f"Designer folder does not exist: {designer_folder}")
            return None
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        image_files = sorted([f for f in folder_path.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions], key=lambda p: p.name)
        
        if not image_files:
            logger.error(f"No images found in designer folder: {designer_folder}")
            return None
        
        logger.info(f"Found {len(image_files)} images")
        
        # Analyze each image
        individual_analyses = []
        analyzed_files = []
        total_images = len(image_files)
        
        for i, image_file in enumerate(image_files):
            if progress_callback:
                progress = 0.3 + (i / total_images) * 0.5  # 30%-80% for image analysis
                progress_callback(progress, f"Analyzing image {i+1}/{total_images}: {image_file.name}")
            
            logger.info(f"Analyzing image {i+1}/{total_images}: {image_file.name}")
            
            analysis = self.analyze_single_image(str(image_file))
            if analysis:
                try:
                    validate_profile(json.loads(analysis))
                except (ValueError, TypeError):
                    logger.error("Invalid nine-field observation; stopping without fallback")
                    return None
                individual_analyses.append(analysis)
                analyzed_files.append(image_file.name)
            
            # Add delay to avoid API rate limiting
            if i < len(image_files) - 1:
                time.sleep(2)
        
        if not individual_analyses:
            logger.error("No images were successfully analyzed")
            return None
        
        if progress_callback:
            progress_callback(0.8, f"Image analysis completed, starting semantic distillation")
        
        logger.info(f"Successfully analyzed {len(individual_analyses)} images, starting semantic distillation")
        
        # Semantic distillation
        style_summary = self.synthesize_style(individual_analyses)
        if not style_summary:
            logger.error("Semantic distillation failed")
            return None
        
        # Build final result
        final_result = {
            "designer_info": {
                "name": designer_name,
                "hash": designer_hash,
                "total_images": len(image_files),
                "analyzed_images": len(individual_analyses),
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "individual_analyses": individual_analyses,
            "individual_analysis_sources": [
                {"image_id": f"image_{i+1}", "filename": name}
                for i, name in enumerate(analyzed_files)
            ],
            "style_analysis": style_summary,
            "metadata": {
                "analyzer_version": SCHEMA_VERSION,
                "aggregation_version": AGGREGATION_VERSION,
                "analysis_dimensions": 9
            }
        }
        
        logger.info(f"Designer style analysis completed: {designer_name}")
        return final_result
    
    def save_analysis_result(self, result: Dict[str, Any], output_path: str, model_name: str = None) -> bool:
        """Save analysis result and training summary"""
        try:
            # Save complete analysis result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Complete analysis result saved: {output_path}")
            
            # Create training summary for design usage
            training_summary = {
                "designer_info": {
                    "name": result["designer_info"]["name"],
                    "hash": result["designer_info"]["hash"],
                    "total_images": result["designer_info"]["total_images"],
                    "analyzed_images": result["designer_info"]["analyzed_images"],
                    "analysis_date": result["designer_info"]["analysis_date"]
                },
                "style_summary": result["style_analysis"],
                "model_info": {
                    "model_name": model_name or "unknown",
                    "vlm_model": self.vlm_client.model_name,
                    "llm_model": self.llm_client.model_name
                },
                "metadata": {
                    "analyzer_version": result["metadata"]["analyzer_version"],
                    "analysis_dimensions": result["metadata"]["analysis_dimensions"],
                    "aggregation_version": result["metadata"].get("aggregation_version"),
                    "observation_sources": result.get("individual_analysis_sources", []),
                    "file_type": "training_summary"
                }
            }
            
            # Save training summary to train.json
            train_file_path = output_path.replace("style_analysis.json", "train.json")
            with open(train_file_path, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Training summary saved: {train_file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis result: {str(e)}")
            return False
