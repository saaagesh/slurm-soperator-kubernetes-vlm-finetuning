#!/usr/bin/env python3
"""
Simple Inference Server - Based on llama32_11b_vlm.ipynb
Clean implementation following the exact notebook approach
"""

import os
import sys
import torch
import copy
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import requests

# Setup environment
os.environ["HF_HOME"] = "/mnt/jail/mnt/models/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/jail/mnt/models/cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# Global variables
base_model = None
model_ft = None
processor = None
models_loaded = {"base": False, "finetuned": False}
loading_status = {"base": "Loading...", "finetuned": "Loading..."}

def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_fine_tune_model(base_model_id="meta-llama/Llama-3.2-11B-Vision", adapter_path=None):
    """
    Load models exactly as in the notebook
    """
    global base_model, model_ft, processor, models_loaded, loading_status
    
    try:
        log_with_timestamp("Loading models using notebook approach...")
        
        # Handle authentication
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                log_with_timestamp("✓ Successfully authenticated with Hugging Face")
            except Exception as auth_error:
                log_with_timestamp(f"⚠️  HF authentication failed: {auth_error}")
                log_with_timestamp("Continuing without authentication")
        
        # Determine adapter path
        if not adapter_path:
            adapter_path = os.environ.get('FINE_TUNED_MODEL_PATH')
            if not adapter_path:
                if os.path.exists('/mnt/jail/mnt/jail/mnt/models/fine-tuned'):
                    adapter_path = '/mnt/jail/mnt/jail/mnt/models/fine-tuned'
                elif os.path.exists('/mnt/jail/mnt/models/fine-tuned'):
                    adapter_path = '/mnt/jail/mnt/models/fine-tuned'
                else:
                    raise Exception("Fine-tuned model path not found")
        
        log_with_timestamp(f"Using adapter path: {adapter_path}")
        
        # Load processor (from notebook)
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        loading_status["base"] = "Loading processor..."
        processor = AutoProcessor.from_pretrained(base_model_id)
        log_with_timestamp("✓ Processor loaded")
        
        # Load base model (from notebook)
        loading_status["base"] = "Loading base model..."
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16  # Using float16 as in notebook
        )
        models_loaded["base"] = True
        loading_status["base"] = "Ready"
        log_with_timestamp("✓ Base model loaded")
        
        # Create fine-tuned model (from notebook approach)
        loading_status["finetuned"] = "Creating fine-tuned model..."
        model_ft = copy.deepcopy(base_model)
        log_with_timestamp("✓ Created deep copy of base model")
        
        # Load adapter (from notebook)
        loading_status["finetuned"] = "Loading adapter..."
        model_ft.load_adapter(adapter_path)
        models_loaded["finetuned"] = True
        loading_status["finetuned"] = "Ready"
        log_with_timestamp("✓ Fine-tuned model loaded with adapter")
        
        log_with_timestamp("✓ All models loaded successfully using notebook approach!")
        
    except Exception as e:
        log_with_timestamp(f"✗ Error loading models: {e}")
        models_loaded["base"] = False
        models_loaded["finetuned"] = False
        loading_status["base"] = f"Error: {str(e)}"
        loading_status["finetuned"] = f"Error: {str(e)}"
        import traceback
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}")

def generate(model, sample, processor):
    """
    Generate function exactly from the notebook
    """
    # Prompt template from notebook
    prompt = """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""
    
    system_message = "You are an expert product description writer for Amazon."
    
    # Format prompt exactly as in notebook
    formatted_prompt = prompt.format(product_name=sample["Product Name"], category=sample["Category"])
    formatted_prompt = ("<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>{formatted_prompt}<|image|><|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
    )
    
    image = sample['image']
    
    # Handle URL images (from notebook)
    if isinstance(image, str) and image.startswith("https://"):
        response = requests.get(image)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Process inputs (from notebook)
    inputs = processor(images=[image], text=formatted_prompt, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # Generate (from notebook)
    generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.8, top_p=0.9, do_sample=True)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def preprocess_image(image_b64_data):
    """Preprocess base64 image data"""
    try:
        image_data = base64.b64decode(image_b64_data)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        log_with_timestamp(f"Image preprocessing error: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

# Flask routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if all(models_loaded.values()) else "partial",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "base": {
                "loaded": models_loaded["base"],
                "status": loading_status["base"]
            },
            "finetuned": {
                "loaded": models_loaded["finetuned"],
                "status": loading_status["finetuned"]
            }
        }
    })

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        "server": "Simple Inference Server",
        "version": "1.0",
        "approach": "Notebook-based with copy.deepcopy() and load_adapter()",
        "models": {
            "base": {
                "loaded": models_loaded["base"],
                "status": loading_status["base"]
            },
            "finetuned": {
                "loaded": models_loaded["finetuned"],
                "status": loading_status["finetuned"]
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/generate/base', methods=['POST'])
def generate_base():
    """Generate with base model"""
    if not models_loaded["base"]:
        return jsonify({
            "error": f"Base model not ready: {loading_status['base']}",
            "model": "base",
            "status": "error"
        }), 503
    
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data", "status": "error"}), 400
        
        # Preprocess image
        image = preprocess_image(data['image'])
        
        # Create sample in notebook format
        sample = {
            "Product Name": data.get('product_name', 'Unknown Product'),
            "Category": data.get('category', 'General'),
            "image": image
        }
        
        # Generate using notebook function
        description = generate(base_model, sample, processor)
        
        return jsonify({
            "description": description,
            "model": "base",
            "status": "success"
        })
        
    except Exception as e:
        log_with_timestamp(f"Error in base generation: {e}")
        return jsonify({
            "error": str(e),
            "model": "base",
            "status": "error"
        }), 500

@app.route('/generate/finetuned', methods=['POST'])
def generate_finetuned():
    """Generate with fine-tuned model"""
    if not models_loaded["finetuned"]:
        return jsonify({
            "error": f"Fine-tuned model not ready: {loading_status['finetuned']}",
            "model": "finetuned",
            "status": "error"
        }), 503
    
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data", "status": "error"}), 400
        
        # Preprocess image
        image = preprocess_image(data['image'])
        
        # Create sample in notebook format
        sample = {
            "Product Name": data.get('product_name', 'Unknown Product'),
            "Category": data.get('category', 'General'),
            "image": image
        }
        
        # Generate using notebook function
        description = generate(model_ft, sample, processor)
        
        return jsonify({
            "description": description,
            "model": "finetuned",
            "status": "success"
        })
        
    except Exception as e:
        log_with_timestamp(f"Error in fine-tuned generation: {e}")
        return jsonify({
            "error": str(e),
            "model": "finetuned",
            "status": "error"
        }), 500

@app.route('/generate/both', methods=['POST'])
def generate_both():
    """Generate with both models"""
    if not models_loaded["base"] or not models_loaded["finetuned"]:
        missing_models = []
        if not models_loaded["base"]:
            missing_models.append(f"base: {loading_status['base']}")
        if not models_loaded["finetuned"]:
            missing_models.append(f"finetuned: {loading_status['finetuned']}")
        
        return jsonify({
            "error": f"Models not ready - {', '.join(missing_models)}",
            "status": "error"
        }), 503
    
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data", "status": "error"}), 400
        
        # Preprocess image
        image = preprocess_image(data['image'])
        
        # Create sample in notebook format
        sample = {
            "Product Name": data.get('product_name', 'Unknown Product'),
            "Category": data.get('category', 'General'),
            "image": image
        }
        
        # Generate with both models
        base_description = generate(base_model, sample, processor)
        finetuned_description = generate(model_ft, sample, processor)
        
        return jsonify({
            "descriptions": {
                "base": base_description,
                "finetuned": finetuned_description
            },
            "status": "success"
        })
        
    except Exception as e:
        log_with_timestamp(f"Error in both generation: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/', methods=['GET'])
def serve_ui():
    """Serve simple UI"""
    return """
    <html><body>
    <h1>Simple Llama 3.2 VLM Inference Server</h1>
    <p>Based on notebook approach with copy.deepcopy() and load_adapter()</p>
    <p>Server is running! API endpoints available:</p>
    <ul>
        <li><a href="/health">/health</a> - Server health check</li>
        <li><a href="/status">/status</a> - Detailed status</li>
        <li>POST /generate/base - Generate with base model</li>
        <li>POST /generate/finetuned - Generate with fine-tuned model</li>
        <li>POST /generate/both - Generate with both models</li>
    </ul>
    </body></html>
    """, 200, {'Content-Type': 'text/html'}

def main():
    """Main function"""
    log_with_timestamp("Starting Simple Inference Server...")
    log_with_timestamp("Using notebook approach: copy.deepcopy() + load_adapter()")
    
    # Print system info
    log_with_timestamp(f"Python version: {sys.version}")
    log_with_timestamp(f"PyTorch version: {torch.__version__}")
    log_with_timestamp(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_with_timestamp(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log_with_timestamp(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Start model loading in thread
    model_thread = threading.Thread(target=load_fine_tune_model, daemon=True)
    log_with_timestamp("Starting model loading thread...")
    model_thread.start()
    
    # Start Flask server
    log_with_timestamp("Starting Flask server on port 5000...")
    log_with_timestamp("Endpoints available:")
    log_with_timestamp("  GET  /                - Simple UI")
    log_with_timestamp("  GET  /health          - Health check")
    log_with_timestamp("  GET  /status          - Status")
    log_with_timestamp("  POST /generate/base   - Generate with base model")
    log_with_timestamp("  POST /generate/finetuned - Generate with fine-tuned model")
    log_with_timestamp("  POST /generate/both   - Generate with both models")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()