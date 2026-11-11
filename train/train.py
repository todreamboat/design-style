import gradio as gr
import os
import json
import hashlib
import shutil
from typing import List, Optional

# Ensure cache directory exists
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(cache_dir, exist_ok=True)

# designer.json file path
designer_json_path = os.path.join(cache_dir, "designer.json")

def load_designers():
    """Load existing designer data"""
    if os.path.exists(designer_json_path):
        try:
            with open(designer_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_designers(designers_data):
    """Save designer data"""
    with open(designer_json_path, 'w', encoding='utf-8') as f:
        json.dump(designers_data, f, ensure_ascii=False, indent=2)

def generate_hash(designer_name: str) -> str:
    """Generate unique hash based on designer name and current time"""
    import time
    unique_string = f"{designer_name}_{int(time.time() * 1000000)}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:16]

def upload_images(designer_name: str, images: List) -> tuple:
    """Handle image upload"""
    if not designer_name or not designer_name.strip():
        return "Please enter designer name!", None, None, None, gr.update(), gr.update(), gr.update(), gr.update()
    
    if not images or len(images) == 0:
        return "Please upload at least one image!", None, None, None, gr.update(), gr.update(), gr.update(), gr.update()
    
    if len(images) > 30:
        return "Maximum 30 images allowed!", None, None, None, gr.update(), gr.update(), gr.update(), gr.update()
    
    designer_name = designer_name.strip()
    
    # Load existing designer data
    designers_data = load_designers()
    
    # Check if designer name already exists
    for existing_hash, existing_name in designers_data.items():
        if existing_name == designer_name:
            return f"Designer name '{designer_name}' already exists! Please use a different name.", None, None, None, gr.update(), gr.update(), gr.update(), gr.update()
    
    # Generate unique hash
    designer_hash = generate_hash(designer_name)
    
    # Ensure hash is unique
    while designer_hash in designers_data:
        designer_hash = generate_hash(designer_name)
    
    # Create designer folder
    designer_folder = os.path.join(cache_dir, designer_hash)
    os.makedirs(designer_folder, exist_ok=True)
    
    try:
        # Copy images to designer folder
        for i, image in enumerate(images):
            if image is not None:
                # Get image file extension
                file_extension = os.path.splitext(image.name)[-1] if hasattr(image, 'name') and image.name else '.jpg'
                # Target filename
                target_filename = f"image_{i+1:03d}{file_extension}"
                target_path = os.path.join(designer_folder, target_filename)
                
                # Copy file
                if hasattr(image, 'name'):
                    # If it's a file object
                    shutil.copy2(image.name, target_path)
                else:
                    # If it's other format
                    shutil.copy2(image, target_path)
        
        # Update designer data
        designers_data[designer_hash] = designer_name
        save_designers(designers_data)
        
        success_message = f"Upload completed!\nDesigner: {designer_name}\nHash: {designer_hash}\nUploaded {len(images)} images"
        
        # Return success message and locked controls
        return success_message, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=True)
        
    except Exception as e:
        # Clean up created folder if error occurs
        if os.path.exists(designer_folder):
            shutil.rmtree(designer_folder)
        return f"Upload failed: {str(e)}", None, None, None, gr.update(), gr.update(), gr.update(), gr.update()

def train_model(selected_model, progress=gr.Progress()):
    """Train model"""
    try:
        # Import necessary modules
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from config.settings import get_settings
        from core.style_analyzer import StyleAnalyzer
        from loguru import logger
        
        # Configure logging
        logger.add("logs/training.log", rotation="1 week", retention="1 month")
        
        progress(0.1, desc="Initializing configuration...")
        
        settings = get_settings()

        if not settings.api_vlm_api_key or not settings.api_llm_api_key:
            return (
                "API credentials are not configured. Set "
                "DST_API_VLM_API_KEY and DST_API_LLM_API_KEY in the runtime environment."
            )
        
        # Load designer data
        designers_data = load_designers()
        if not designers_data:
            return "No uploaded designer data found! Please upload designer images first."
        
        # Get latest uploaded designer (assume it's the last one)
        latest_hash = list(designers_data.keys())[-1]
        latest_designer = designers_data[latest_hash]
        
        progress(0.2, desc=f"Preparing to analyze designer: {latest_designer}")
        
        logger.info(f"Starting training for designer: {latest_designer} ({latest_hash})")
        
        # Create style analyzer with selected model
        vlm_config = {
            "api_endpoint": settings.api_vlm_api_endpoint,
            "api_key": settings.api_vlm_api_key,
            "model_name": selected_model  # Use selected model
        }
        
        llm_config = {
            "api_endpoint": settings.api_llm_api_endpoint,
            "api_key": settings.api_llm_api_key,
            "model_name": selected_model  # Use selected model
        }
        
        analyzer = StyleAnalyzer(vlm_config, llm_config)
        
        # Designer folder path
        designer_folder = os.path.join(cache_dir, latest_hash)
        
        progress(0.3, desc="Starting style analysis...")
        
        # Create progress callback function
        def progress_callback(progress_value, desc):
            progress(progress_value, desc=desc)
        
        # Analyze designer style
        logger.info("Performing style analysis...")
        analysis_result = analyzer.analyze_designer_style(
            designer_folder=designer_folder,
            designer_name=latest_designer,
            designer_hash=latest_hash,
            progress_callback=progress_callback
        )
        
        progress(0.9, desc="Saving analysis results...")
        
        if not analysis_result:
            return "Style analysis failed! Please check images or API configuration."
        
        # Save analysis results with model information
        result_file = os.path.join(designer_folder, "style_analysis.json")
        if analyzer.save_analysis_result(analysis_result, result_file, selected_model):
            progress(1.0, desc="Training completed!")
            
            success_msg = f"✅ Training completed!\n\n"
            success_msg += f"🎨 Designer: {latest_designer}\n"
            success_msg += f"🔑 Hash: {latest_hash}\n"
            success_msg += f"🤖 Model: {selected_model}\n"
            success_msg += f"📊 Analyzed images: {analysis_result['designer_info']['analyzed_images']}\n"
            success_msg += f"📁 Total images: {analysis_result['designer_info']['total_images']}\n"
            success_msg += f"💾 Analysis results saved to style_analysis.json\n"
            success_msg += f"🎯 Training summary saved to train.json\n"
            
            # Add style summary
            if 'style_analysis' in analysis_result and 'overall_style_description' in analysis_result['style_analysis']:
                success_msg += f"\n� Designer style characteristics:\n{analysis_result['style_analysis']['overall_style_description'][:300]}..."
            
            logger.info(f"Training completed: {latest_designer}")
            return success_msg
        else:
            return "Training completed but failed to save results!"
            
    except ImportError as e:
        error_msg = f"❌ Module import failed: {str(e)}\nPlease ensure all dependencies are installed."
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error occurred during training: {str(e)}"
        logger.error(error_msg)
        return error_msg

def reset_interface():
    """Reset interface"""
    return "", "gpt-4o", None, "Please enter designer name and upload images", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False)

# Create Gradio interface
with gr.Blocks(title="Meta Designer", theme=gr.themes.Soft()) as app:
    # Top title
    gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="font-size: 3em; color: #2E86AB; margin: 0;">Meta Designer</h1>
            <p style="font-size: 1.2em; color: #666; margin: 10px 0;">Training Assistant</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            # Designer name input box
            designer_name_input = gr.Textbox(
                label="Designer Name",
                placeholder="Enter designer name...",
                interactive=True
            )
            
            # Model selection dropdown
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=[
                    "qwen/qwen3.8-max-0902",
                    "google/gemini-3.1-pro-preview",
                    "google/gemini-2.5-pro", 
                    "qwen/qwen2.5-vl-72b-instruct",
                    "openai/gpt-4o"
                ],
                value="qwen/qwen3.8-max-0902",
                interactive=True,
                info="Choose the AI model for style analysis"
            )
            
            # Image upload box
            image_upload = gr.File(
                label="Upload Design Works",
                file_count="multiple",
                file_types=["image"],
                interactive=True
            )
            
            # Image preview
            with gr.Row():
                image_gallery = gr.Gallery(
                    label="Preview Uploaded Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=3,
                    height="auto"
                )
    
    # Status display
    status_output = gr.Textbox(
        label="Status",
        value="Please enter designer name and upload images",
        interactive=False,
        lines=3
    )
    
    # Button area
    with gr.Row():
        upload_btn = gr.Button("Upload", variant="primary", size="lg")
        train_btn = gr.Button("Train", variant="secondary", size="lg", interactive=False)
        reset_btn = gr.Button("Reset", variant="stop", size="lg", visible=False)
    
    # Auto preview when images are uploaded
    def update_gallery(files):
        if files:
            # Limit display to maximum 30 images
            display_files = files[:30] if len(files) > 30 else files
            return display_files
        return None
    
    image_upload.change(
        fn=update_gallery,
        inputs=[image_upload],
        outputs=[image_gallery]
    )
    
    # Upload button click event
    upload_btn.click(
        fn=upload_images,
        inputs=[designer_name_input, image_upload],
        outputs=[status_output, designer_name_input, model_dropdown, image_upload, train_btn, reset_btn]
    )
    
    # Train button click event
    train_btn.click(
        fn=train_model,
        inputs=[model_dropdown],
        outputs=[status_output]
    )
    
    # Reset button click event
    reset_btn.click(
        fn=reset_interface,
        inputs=[],
        outputs=[designer_name_input, model_dropdown, image_upload, status_output, designer_name_input, model_dropdown, image_upload, train_btn, reset_btn]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=12002,
        share=False,
        show_error=True
    )
