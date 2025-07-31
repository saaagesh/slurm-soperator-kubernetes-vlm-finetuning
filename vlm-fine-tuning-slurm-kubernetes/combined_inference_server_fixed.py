#!/usr/bin/env python3
"""
Optimized Combined Inference Server - Complete Implementation
Enhanced version with better error handling, caching, and performance optimizations
"""

import os
import sys
import warnings
import torch
import copy
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import json
from functools import lru_cache
import gc
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup environment - use your exact jail mount paths
os.environ["HF_HOME"] = "/mnt/jail/mnt/models/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/jail/mnt/models/cache/transformers"

# Performance optimizations
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
torch.set_float32_matmul_precision('high')  # Better performance

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# Global variables for models and processors
base_model = None
finetuned_model = None
base_processor = None
finetuned_processor = None

# Model loading status
models_loaded = {"base": False, "finetuned": False}
loading_status = {"base": "Loading...", "finetuned": "Loading..."}
loading_start_time = {"base": None, "finetuned": None}

# Performance tracking
generation_stats = {
    "base": {"count": 0, "total_time": 0, "avg_time": 0},
    "finetuned": {"count": 0, "total_time": 0, "avg_time": 0}
}

def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def setup_base_model():
    """Initialize the base Llama 3.2 11B VLM model with optimizations"""
    global base_model, base_processor, models_loaded, loading_status, loading_start_time
    
    try:
        loading_start_time["base"] = time.time()
        log_with_timestamp("Loading base Llama 3.2 11B VLM model...")
        loading_status["base"] = "Loading base model..."
        
        # Use the exact imports from the notebook
        from transformers import AutoProcessor as MllamaProcessor, AutoModelForVision2Seq as MllamaForConditionalGeneration
        
        from huggingface_hub import login
        
        # Handle authentication with better error handling and debugging
        hf_token = os.environ.get('HF_TOKEN')
        log_with_timestamp(f"HF_TOKEN environment variable: {'SET' if hf_token else 'NOT SET'}")
        if hf_token:
            log_with_timestamp(f"HF_TOKEN starts with: {hf_token[:10]}...")
            try:
                # Clear any existing token cache that might be causing issues
                from huggingface_hub import logout
                try:
                    logout()
                    log_with_timestamp("Cleared existing HF authentication")
                except:
                    pass  # Ignore if no existing auth
                
                login(token=hf_token)
                log_with_timestamp("✓ Successfully authenticated with Hugging Face")
            except Exception as auth_error:
                log_with_timestamp(f"⚠️  HF authentication failed: {auth_error}")
                log_with_timestamp("Continuing without authentication - may cause issues with gated models")
        
        MODEL_ID = "meta-llama/Llama-3.2-11B-Vision"
        
        # Enhanced quantization setup
        quantization_config = None
        use_quantization = False
        
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16  # Better storage efficiency
            )
            use_quantization = True
            log_with_timestamp("✓ Using enhanced 4-bit quantization for base model")
            
        except Exception as e:
            log_with_timestamp(f"⚠️  Bitsandbytes not working for base model: {e}")
            log_with_timestamp("Loading base model without quantization")
            use_quantization = False
        
        # Load processor with caching - match notebook exactly
        loading_status["base"] = "Loading base processor..."
        base_processor = MllamaProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        
        # Load model with optimizations
        loading_status["base"] = "Loading base model weights..."
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # Memory optimization
            "use_safetensors": True     # Faster loading
        }
        
        if use_quantization:
            model_kwargs["quantization_config"] = quantization_config
        
        base_model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            **model_kwargs
        )
        
        # Model optimizations
        base_model.eval()  # Set to evaluation mode
        if hasattr(base_model, 'compile'):
            try:
                base_model = torch.compile(base_model, mode="reduce-overhead")
                log_with_timestamp("✓ Base model compiled with torch.compile")
            except Exception as e:
                log_with_timestamp(f"⚠️  Could not compile base model: {e}")
        
        loading_time = time.time() - loading_start_time["base"]
        models_loaded["base"] = True
        loading_status["base"] = "Ready"
        log_with_timestamp(f"✓ Base model loaded successfully in {loading_time:.1f}s!")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        models_loaded["base"] = False
        loading_status["base"] = f"Error: {str(e)}"
        log_with_timestamp(f"✗ Error loading base model: {e}")

def setup_finetuned_model():
    """Initialize the fine-tuned Llama 3.2 11B VLM model with enhanced error handling"""
    global finetuned_model, finetuned_processor, models_loaded, loading_status, loading_start_time
    
    try:
        loading_start_time["finetuned"] = time.time()
        log_with_timestamp("Loading fine-tuned Llama 3.2 11B VLM model...")
        loading_status["finetuned"] = "Loading fine-tuned model..."
        
        # Use the exact imports from the notebook
        from transformers import AutoProcessor as MllamaProcessor, AutoModelForVision2Seq as MllamaForConditionalGeneration
        
        from huggingface_hub import login
        
        # Handle authentication with better error handling and debugging
        hf_token = os.environ.get('HF_TOKEN')
        log_with_timestamp(f"HF_TOKEN for fine-tuned model: {'SET' if hf_token else 'NOT SET'}")
        if hf_token:
            log_with_timestamp(f"HF_TOKEN starts with: {hf_token[:10]}...")
            try:
                # Clear any existing token cache that might be causing issues
                from huggingface_hub import logout
                try:
                    logout()
                    log_with_timestamp("Cleared existing HF authentication for fine-tuned model")
                except:
                    pass  # Ignore if no existing auth
                
                login(token=hf_token)
                log_with_timestamp("✓ Successfully authenticated with Hugging Face for fine-tuned model")
            except Exception as auth_error:
                log_with_timestamp(f"⚠️  HF authentication failed for fine-tuned model: {auth_error}")
                log_with_timestamp("Continuing without authentication - may cause issues with gated models")
        
        BASE_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision"
        
        # Get adapter path with fallback options
        ADAPTER_PATH = os.environ.get('FINE_TUNED_MODEL_PATH')
        
        # If not set via environment, try both possible locations
        if not ADAPTER_PATH:
            if os.path.exists('/mnt/jail/mnt/jail/mnt/models/fine-tuned'):
                ADAPTER_PATH = '/mnt/jail/mnt/jail/mnt/models/fine-tuned'
                log_with_timestamp("Using primary model path: /mnt/jail/mnt/jail/mnt/models/fine-tuned")
            elif os.path.exists('/mnt/jail/mnt/models/fine-tuned'):
                ADAPTER_PATH = '/mnt/jail/mnt/models/fine-tuned'
                log_with_timestamp("Using fallback model path: /mnt/jail/mnt/models/fine-tuned")
            else:
                raise Exception("Fine-tuned model not found at either location: /mnt/jail/mnt/jail/mnt/models/fine-tuned or /mnt/jail/mnt/models/fine-tuned")
        
        # Verify the final path exists
        if not os.path.exists(ADAPTER_PATH):
            raise Exception(f"Fine-tuned model not found at: {ADAPTER_PATH}")
        
        # Validate adapter files
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        for file in required_files:
            if not os.path.exists(os.path.join(ADAPTER_PATH, file)):
                log_with_timestamp(f"⚠️  Missing file: {file}")
        
        log_with_timestamp(f"Using fine-tuned model from: {ADAPTER_PATH}")
        
        # Enhanced quantization for fine-tuned model
        quantization_config = None
        use_quantization = False
        
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16
            )
            use_quantization = True
            log_with_timestamp("✓ Using enhanced 4-bit quantization for fine-tuned model")
            
        except Exception as e:
            log_with_timestamp(f"⚠️  Bitsandbytes not working for fine-tuned model: {e}")
            use_quantization = False
        
        # Load processor (shared with base model to save memory) - match notebook exactly
        loading_status["finetuned"] = "Loading fine-tuned processor..."
        finetuned_processor = MllamaProcessor.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True
        )
        
        # Load base model for fine-tuning with proper device handling
        loading_status["finetuned"] = "Loading base model for fine-tuning..."
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        
        # Don't use device_map="auto" for PEFT models to avoid meta tensor issues
        if use_quantization:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"  # Only use device_map with quantization
        
        base_model_ft = MllamaForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            **model_kwargs
        )
        
        # If model is on meta device, move it properly
        if not use_quantization:
            try:
                first_param = next(base_model_ft.parameters())
                if hasattr(first_param, 'device') and str(first_param.device) == 'meta':
                    log_with_timestamp("Model loaded on meta device, moving to CUDA using to_empty()")
                    base_model_ft = base_model_ft.to_empty(device='cuda')
                elif not use_quantization:
                    # Move to CUDA if not using quantization
                    base_model_ft = base_model_ft.cuda()
                    log_with_timestamp("Moved base model to CUDA")
            except Exception as device_e:
                log_with_timestamp(f"Device handling: {device_e}")
        
        # Enhanced adapter loading with multiple fallback methods
        loading_status["finetuned"] = "Loading LoRA adapters with enhanced error handling..."
        
        finetuned_model = None
        loading_method = None
        
        # Method 0: Pre-check and fix config file if needed
        try:
            import json
            config_path = os.path.join(ADAPTER_PATH, 'adapter_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Check for problematic keys and fix them
                problematic_attrs = ['corda_config', 'cord_config', 'corda', 'cord', 'eva_config']
                fixed_config = False
                for attr in problematic_attrs:
                    if attr in config_dict:
                        log_with_timestamp(f"Found problematic config key: {attr}, removing...")
                        del config_dict[attr]
                        fixed_config = True
                
                # Save cleaned config if we made changes
                if fixed_config:
                    with open(config_path, 'w') as f:
                        json.dump(config_dict, f, indent=2)
                    log_with_timestamp("✓ Config file cleaned and saved")
        except Exception as e:
            log_with_timestamp(f"Config pre-check failed (non-critical): {e}")
        
        # Method 1: Try AutoPeftModel (recommended for newer PEFT versions)
        try:
            from peft import AutoPeftModelForCausalLM
            log_with_timestamp("Attempting Method 1: AutoPeftModelForCausalLM")
            
            finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
                ADAPTER_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            loading_method = "AutoPeftModelForCausalLM"
            log_with_timestamp("✓ Method 1 successful: AutoPeftModelForCausalLM")
            
        except Exception as e1:
            log_with_timestamp(f"Method 1 failed: {e1}")
            
            # Method 2: Try minimal PeftModel.from_pretrained
            try:
                from peft import PeftModel
                log_with_timestamp("Attempting Method 2: Minimal PeftModel.from_pretrained")
                
                finetuned_model = PeftModel.from_pretrained(
                    base_model_ft,
                    ADAPTER_PATH,
                    is_trainable=False
                )
                loading_method = "PeftModel (minimal)"
                log_with_timestamp("✓ Method 2 successful: Minimal PeftModel")
                
            except Exception as e2:
                log_with_timestamp(f"Method 2 failed: {e2}")
                
                # Method 3: Manual config loading with cleanup
                try:
                    from peft import PeftConfig, PeftModel
                    import json
                    log_with_timestamp("Attempting Method 3: Manual config cleanup")
                    
                    # Load and clean config file manually
                    config_path = os.path.join(ADAPTER_PATH, 'adapter_config.json')
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    log_with_timestamp(f"Original config keys: {list(config_dict.keys())}")
                    
                    # Remove any problematic attributes from config dict
                    problematic_attrs = ['corda_config', 'cord_config', 'corda', 'cord', 'eva_config']
                    for attr in problematic_attrs:
                        if attr in config_dict:
                            del config_dict[attr]
                            log_with_timestamp(f"Removed problematic attribute: {attr}")
                    
                    # Create clean config using LoraConfig directly
                    from peft import LoraConfig
                    config = LoraConfig(
                        r=config_dict.get('r', 8),
                        lora_alpha=config_dict.get('lora_alpha', 16),
                        target_modules=config_dict.get('target_modules', ["q_proj", "k_proj", "v_proj"]),
                        lora_dropout=config_dict.get('lora_dropout', 0.05),
                        bias=config_dict.get('bias', "none"),
                        task_type=config_dict.get('task_type', "CAUSAL_LM"),
                        inference_mode=True
                    )
                    
                    finetuned_model = PeftModel.from_pretrained(
                        base_model_ft,
                        ADAPTER_PATH,
                        config=config,
                        is_trainable=False
                    )
                    loading_method = "PeftModel (manual config)"
                    log_with_timestamp("✓ Method 3 successful: Manual config cleanup")
                    
                except Exception as e3:
                    log_with_timestamp(f"Method 3 failed: {e3}")
                    
                    # Method 4: Create fresh config and load weights manually
                    try:
                        from peft import LoraConfig, get_peft_model
                        from safetensors.torch import load_file
                        log_with_timestamp("Attempting Method 4: Fresh config + manual weight loading")
                        
                        # Read original config to get proper parameters
                        config_path = os.path.join(ADAPTER_PATH, 'adapter_config.json')
                        with open(config_path, 'r') as f:
                            orig_config = json.load(f)
                        
                        # Create a completely new LoRA config with original parameters
                        lora_config = LoraConfig(
                            r=orig_config.get('r', 8),
                            lora_alpha=orig_config.get('lora_alpha', 16),
                            target_modules=orig_config.get('target_modules', ["q_proj", "k_proj", "v_proj"]),
                            lora_dropout=orig_config.get('lora_dropout', 0.05),
                            bias=orig_config.get('bias', "none"),
                            task_type=orig_config.get('task_type', "CAUSAL_LM"),
                            inference_mode=True
                        )
                        
                        # Ensure base model is properly initialized and not on meta device
                        try:
                            # Check if model is on meta device
                            first_param = next(base_model_ft.parameters())
                            if hasattr(first_param, 'device'):
                                if first_param.device.type == 'meta' or str(first_param.device) == 'meta':
                                    log_with_timestamp("Model is on meta device, using to_empty() to move to cuda")
                                    base_model_ft = base_model_ft.to_empty(device='cuda')
                                    log_with_timestamp("Successfully moved model from meta to CUDA")
                                else:
                                    log_with_timestamp(f"Model already on device: {first_param.device}")
                            else:
                                log_with_timestamp("Could not determine model device, attempting to move to CUDA")
                                base_model_ft = base_model_ft.cuda()
                        except Exception as device_e:
                            log_with_timestamp(f"Device handling error: {device_e}")
                            # Try alternative approach
                            try:
                                base_model_ft = base_model_ft.cuda()
                                log_with_timestamp("Fallback: moved model to CUDA using .cuda()")
                            except Exception as cuda_e:
                                log_with_timestamp(f"CUDA fallback also failed: {cuda_e}")
                        
                        # Apply LoRA to base model
                        finetuned_model = get_peft_model(base_model_ft, lora_config)
                        
                        # Load the saved adapter weights
                        adapter_weights_path = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
                        if os.path.exists(adapter_weights_path):
                            adapter_weights = load_file(adapter_weights_path)
                            
                            # Move weights to proper device
                            device = next(finetuned_model.parameters()).device
                            adapter_weights = {k: v.to(device) for k, v in adapter_weights.items()}
                            
                            # Load weights with error handling
                            missing_keys, unexpected_keys = finetuned_model.load_state_dict(adapter_weights, strict=False)
                            if missing_keys:
                                log_with_timestamp(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                            if unexpected_keys:
                                log_with_timestamp(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                            
                            loading_method = "Fresh config + manual weights"
                            log_with_timestamp("✓ Method 4 successful: Fresh config + manual weight loading")
                        else:
                            raise Exception("adapter_model.safetensors not found")
                            
                    except Exception as e4:
                        log_with_timestamp(f"Method 4 failed: {e4}")
                        
                        # Method 5: Try loading with ignore_mismatched_sizes
                        try:
                            from peft import PeftModel
                            log_with_timestamp("Attempting Method 5: PeftModel with ignore_mismatched_sizes")
                            
                            finetuned_model = PeftModel.from_pretrained(
                                base_model_ft,
                                ADAPTER_PATH,
                                is_trainable=False,
                                ignore_mismatched_sizes=True
                            )
                            loading_method = "PeftModel (ignore mismatched)"
                            log_with_timestamp("✓ Method 5 successful: PeftModel with ignore_mismatched_sizes")
                            
                        except Exception as e5:
                            log_with_timestamp(f"Method 5 failed: {e5}")
                            
                            # Method 6: Try with manually cleaned config file
                            try:
                                from peft import PeftModel, LoraConfig
                                log_with_timestamp("Attempting Method 6: Direct config creation from saved values")
                                
                                # Read the original config to get the actual values
                                config_path = os.path.join(ADAPTER_PATH, 'adapter_config.json')
                                with open(config_path, 'r') as f:
                                    orig_config = json.load(f)
                                
                                # Create a completely clean LoraConfig with only the essential parameters
                                clean_config = LoraConfig(
                                    r=orig_config.get('r', 8),
                                    lora_alpha=orig_config.get('lora_alpha', 16),
                                    target_modules=orig_config.get('target_modules', ["q_proj", "k_proj", "v_proj"]),
                                    lora_dropout=orig_config.get('lora_dropout', 0.05),
                                    bias=orig_config.get('bias', "none"),
                                    task_type="CAUSAL_LM",  # Force this value
                                    inference_mode=True
                                )
                                
                                # Try loading with the clean config
                                finetuned_model = PeftModel.from_pretrained(
                                    base_model_ft,
                                    ADAPTER_PATH,
                                    config=clean_config,
                                    is_trainable=False
                                )
                                loading_method = "Direct config creation"
                                log_with_timestamp("✓ Method 6 successful: Direct config creation")
                                
                            except Exception as e6:
                                log_with_timestamp(f"Method 6 failed: {e6}")
                                
                                # Method 7: Last resort - try loading just the base model
                                log_with_timestamp("All LoRA loading methods failed. Using base model as fallback.")
                                finetuned_model = base_model_ft
                                loading_method = "Base model fallback"
                                log_with_timestamp("⚠️  Using base model as fine-tuned model (no LoRA applied)")
        
        if finetuned_model is None:
            raise Exception("All loading methods failed")
        
        # Model optimizations
        finetuned_model.eval()
        
        # Only compile if it's not the fallback base model
        if loading_method != "Base model fallback" and hasattr(finetuned_model, 'compile'):
            try:
                finetuned_model = torch.compile(finetuned_model, mode="reduce-overhead")
                log_with_timestamp("✓ Fine-tuned model compiled with torch.compile")
            except Exception as e:
                log_with_timestamp(f"⚠️  Could not compile fine-tuned model: {e}")
        
        loading_time = time.time() - loading_start_time["finetuned"]
        models_loaded["finetuned"] = True
        loading_status["finetuned"] = "Ready"
        log_with_timestamp(f"✓ Fine-tuned model loaded successfully using {loading_method} in {loading_time:.1f}s!")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        models_loaded["finetuned"] = False
        loading_status["finetuned"] = f"Error: {str(e)}"
        log_with_timestamp(f"✗ Error loading fine-tuned model: {e}")
        # Print full traceback for debugging
        import traceback
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}")

@lru_cache(maxsize=32)
def create_optimized_prompt(product_name, category):
    """Create prompt that matches training format exactly"""
    # This MUST match the exact format used in training from the notebook
    prompt_template = """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""
    
    return prompt_template.format(product_name=product_name, category=category)

def preprocess_image(image_b64_data):
    """Preprocess image data with proper error handling"""
    try:
        image_data = base64.b64decode(image_b64_data)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # For VLM models, ensure proper image sizing to avoid CUDA indexing errors
        original_size = image.size
        min_size = 224
        max_size = 512  # Reduced from 1024 to avoid memory issues
        
        # Always resize very small images to a safe minimum size
        if image.size[0] < min_size or image.size[1] < min_size:
            log_with_timestamp(f"Image too small ({image.size}), resizing to {min_size}x{min_size}")
            image = image.resize((min_size, min_size), Image.Resampling.LANCZOS)
        
        # Resize large images more aggressively to avoid CUDA issues
        elif image.size[0] > max_size or image.size[1] > max_size:
            # Resize maintaining aspect ratio but with stricter limits
            aspect_ratio = image.size[0] / image.size[1]
            if aspect_ratio > 1:
                new_size = (max_size, int(max_size / aspect_ratio))
            else:
                new_size = (int(max_size * aspect_ratio), max_size)
            
            # Ensure minimum dimensions
            new_size = (max(new_size[0], min_size), max(new_size[1], min_size))
            
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            log_with_timestamp(f"Resized large image from {original_size} to {new_size}")
        
        # Additional safety check - ensure dimensions are reasonable
        if image.size[0] * image.size[1] > 512 * 512:
            # Force resize to 512x512 if still too large
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            log_with_timestamp(f"Force resized to 512x512 for safety")
        
        # Ensure the image is exactly what the model expects
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        log_with_timestamp(f"Image preprocessed: {image.size}, mode: {image.mode}")
        return image
        
    except Exception as e:
        log_with_timestamp(f"Image preprocessing error: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

def generate_description(model, processor, image, product_name, category, model_type):
    """Enhanced generation with performance tracking"""
    start_time = time.time()
    
    try:
        # Use cached prompt
        base_prompt = create_optimized_prompt(product_name, category)
        
        # Format prompt exactly like in the original notebook
        system_message = "You are an expert product description writer for Amazon."
        
        # Use the exact format from the notebook
        formatted_prompt = ("<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{base_prompt}<|image|><|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>"
        )
        
        # Process image and text with enhanced error handling
        try:
            log_with_timestamp(f"Processing image size: {image.size}, text length: {len(formatted_prompt)}")
            
            # Use the exact format from the notebook
            inputs = processor(
                images=[image], 
                text=formatted_prompt, 
                padding=True, 
                return_tensors="pt"
            )
            
            # Debug input shapes and validate
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    log_with_timestamp(f"Input {key} shape: {value.shape}")
                    
                    # Check for suspicious tensor values that might cause indexing errors
                    if key == 'input_ids':
                        max_id = value.max().item()
                        min_id = value.min().item()
                        log_with_timestamp(f"Token ID range: {min_id} to {max_id}")
                        
                        # Check if any token IDs are out of bounds
                        vocab_size = processor.tokenizer.vocab_size
                        if max_id >= vocab_size:
                            log_with_timestamp(f"WARNING: Token ID {max_id} >= vocab_size {vocab_size}")
                            # Clamp invalid token IDs
                            value = torch.clamp(value, 0, vocab_size - 1)
                            inputs[key] = value
            
            # Move to model device with error checking
            device = next(model.parameters()).device
            log_with_timestamp(f"Moving inputs to device: {device}")
            
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
        except Exception as e:
            log_with_timestamp(f"Error in processor: {e}")
            # Fallback: try with simpler processing
            try:
                log_with_timestamp("Trying fallback with simpler text processing")
                
                # Try with a much simpler prompt using notebook format
                simple_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>Describe this {category} product called {product_name}.<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                
                inputs = processor(
                    images=[image],
                    text=simple_prompt,
                    return_tensors="pt",
                    padding=True
                )
                
                device = next(model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                log_with_timestamp("Fallback processor succeeded with simple prompt")
                
            except Exception as e2:
                log_with_timestamp(f"Fallback processor also failed: {e2}")
                raise e2
        
        # Use exact generation parameters from the notebook
        generation_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Generate response with enhanced error handling
        try:
            log_with_timestamp(f"Starting generation with {model_type} model")
            
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                # Disable autocast for now to avoid potential issues
                generated_ids = model.generate(
                    **inputs,
                    **generation_kwargs
                )
                
            log_with_timestamp(f"Generation completed, output shape: {generated_ids.shape}")
            
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error) or "device-side assert" in str(cuda_error):
                log_with_timestamp(f"CUDA error during generation: {cuda_error}")
                
                # Try to recover by clearing cache and retrying with simpler settings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Return a helpful error message instead of crashing
                error_msg = f"CUDA indexing error occurred. This is likely due to:\n"
                error_msg += f"1. Image size or complexity causing memory issues\n"
                error_msg += f"2. Token sequence length exceeding model limits\n"
                error_msg += f"3. Model configuration mismatch\n"
                error_msg += f"Try with a smaller, simpler image or restart the server."
                
                log_with_timestamp(f"Returning graceful error instead of crash")
                return error_msg
            else:
                raise cuda_error
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Update performance stats
        generation_time = time.time() - start_time
        generation_stats[model_type]["count"] += 1
        generation_stats[model_type]["total_time"] += generation_time
        generation_stats[model_type]["avg_time"] = (
            generation_stats[model_type]["total_time"] / 
            generation_stats[model_type]["count"]
        )
        
        result = output_text[0] if output_text else "No response generated"
        log_with_timestamp(f"{model_type.title()} model generated response in {generation_time:.2f}s")
        
        return result
        
    except Exception as e:
        log_with_timestamp(f"Error generating description with {model_type} model: {e}")
        return f"Error generating description: {str(e)}"
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Enhanced health check with more details
@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**3:.1f}GB",
                "memory_cached": f"{torch.cuda.memory_reserved(i) / 1024**3:.1f}GB",
                "utilization": f"{torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 'N/A'}%"
            }
    
    return jsonify({
        "status": "healthy" if all(models_loaded.values()) else "partial",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "base": {
                "loaded": models_loaded["base"],
                "status": loading_status["base"],
                "stats": generation_stats["base"]
            },
            "finetuned": {
                "loaded": models_loaded["finetuned"],
                "status": loading_status["finetuned"],
                "stats": generation_stats["finetuned"]
            }
        },
        "system": {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_info": gpu_info
        }
    })

# Add performance metrics endpoint
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Performance metrics endpoint"""
    return jsonify({
        "generation_stats": generation_stats,
        "model_status": {
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

# Status endpoint for detailed model information
@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    return jsonify({
        "server": "Combined Inference Server",
        "version": "2.0",
        "models": {
            "base": {
                "loaded": models_loaded["base"],
                "status": loading_status["base"],
                "load_time": f"{time.time() - loading_start_time['base']:.1f}s" if loading_start_time["base"] else "N/A",
                "stats": generation_stats["base"]
            },
            "finetuned": {
                "loaded": models_loaded["finetuned"],
                "status": loading_status["finetuned"],
                "load_time": f"{time.time() - loading_start_time['finetuned']:.1f}s" if loading_start_time["finetuned"] else "N/A",
                "stats": generation_stats["finetuned"]
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/generate/base', methods=['POST'])
def generate_base():
    """Generate product description using base model"""
    if not models_loaded["base"]:
        return jsonify({
            "error": f"Base model not ready: {loading_status['base']}",
            "model": "base",
            "status": "error"
        }), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received", "status": "error"}), 400
        
        # Enhanced input validation
        required_fields = ['image']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}", "status": "error"}), 400
        
        # Decode and validate image
        try:
            image = preprocess_image(data['image'])
        except ValueError as e:
            return jsonify({"error": str(e), "status": "error"}), 400
        
        product_name = data.get('product_name', 'Unknown Product')
        category = data.get('category', 'General')
        
        # Generate description
        description = generate_description(
            base_model, base_processor, image, product_name, category, "base"
        )
        
        return jsonify({
            "description": description,
            "model": "base",
            "status": "success",
            "generation_time": generation_stats["base"]["avg_time"]
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
    """Generate product description using fine-tuned model"""
    if not models_loaded["finetuned"]:
        return jsonify({
            "error": f"Fine-tuned model not ready: {loading_status['finetuned']}",
            "model": "finetuned",
            "status": "error"
        }), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received", "status": "error"}), 400
        
        # Enhanced input validation
        required_fields = ['image']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}", "status": "error"}), 400
        
        # Decode and validate image
        try:
            image = preprocess_image(data['image'])
        except ValueError as e:
            return jsonify({"error": str(e), "status": "error"}), 400
        
        product_name = data.get('product_name', 'Unknown Product')
        category = data.get('category', 'General')
        
        # Generate description
        description = generate_description(
            finetuned_model, finetuned_processor, image, product_name, category, "finetuned"
        )
        
        return jsonify({
            "description": description,
            "model": "finetuned",
            "status": "success",
            "generation_time": generation_stats["finetuned"]["avg_time"]
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
    """Generate product descriptions using both models"""
    # Check if both models are loaded
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
        if not data:
            return jsonify({"error": "No JSON data received", "status": "error"}), 400
        
        # Enhanced input validation
        required_fields = ['image']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}", "status": "error"}), 400
        
        # Decode and validate image
        try:
            image = preprocess_image(data['image'])
        except ValueError as e:
            return jsonify({"error": str(e), "status": "error"}), 400
        
        product_name = data.get('product_name', 'Unknown Product')
        category = data.get('category', 'General')
        
        # Generate descriptions with both models
        base_description = generate_description(
            base_model, base_processor, image, product_name, category, "base"
        )
        
        finetuned_description = generate_description(
            finetuned_model, finetuned_processor, image, product_name, category, "finetuned"
        )
        
        return jsonify({
            "descriptions": {
                "base": base_description,
                "finetuned": finetuned_description
            },
            "status": "success",
            "generation_times": {
                "base": generation_stats["base"]["avg_time"],
                "finetuned": generation_stats["finetuned"]["avg_time"]
            }
        })
        
    except Exception as e:
        log_with_timestamp(f"Error in both generation: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/', methods=['GET'])
def serve_ui():
    """Serve the comparison UI HTML file"""
    try:
        # Try to find the HTML file in common locations
        possible_paths = [
            'comparison_ui.html',
            '/mnt/jail/mnt/models/logs/comparison_ui.html',
            os.path.join(os.path.dirname(__file__), 'comparison_ui.html')
        ]
        
        html_content = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    html_content = f.read()
                break
        
        if html_content:
            return html_content, 200, {'Content-Type': 'text/html'}
        else:
            return """
            <html><body>
            <h1>Llama 3.2 VLM Inference Server</h1>
            <p>Server is running! API endpoints available:</p>
            <ul>
                <li><a href="/health">/health</a> - Server health check</li>
                <li><a href="/status">/status</a> - Detailed status</li>
                <li>POST /generate/base - Generate with base model</li>
                <li>POST /generate/finetuned - Generate with fine-tuned model</li>
                <li>POST /generate/both - Generate with both models</li>
            </ul>
            <p>comparison_ui.html not found in expected locations.</p>
            </body></html>
            """, 200, {'Content-Type': 'text/html'}
            
    except Exception as e:
        return f"Error serving UI: {str(e)}", 500

def main():
    """Enhanced main function"""
    log_with_timestamp("Starting Optimized Combined Inference Server...")
    log_with_timestamp("This server hosts both base and fine-tuned models with performance optimizations")
    
    # Print system info
    log_with_timestamp(f"Python version: {sys.version}")
    log_with_timestamp(f"PyTorch version: {torch.__version__}")
    log_with_timestamp(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_with_timestamp(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log_with_timestamp(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Start model loading in separate threads
    base_thread = threading.Thread(target=setup_base_model, daemon=True)
    finetuned_thread = threading.Thread(target=setup_finetuned_model, daemon=True)
    
    log_with_timestamp("Starting optimized model loading threads...")
    base_thread.start()
    finetuned_thread.start()
    
    # Start Flask server
    log_with_timestamp("Starting Flask server on port 5000...")
    log_with_timestamp("Enhanced endpoints available:")
    log_with_timestamp("  GET  /                - Web UI for model comparison")
    log_with_timestamp("  GET  /health          - Enhanced health check")
    log_with_timestamp("  GET  /metrics         - Performance metrics") 
    log_with_timestamp("  GET  /status          - Detailed status")
    log_with_timestamp("  POST /generate/base   - Generate with base model")
    log_with_timestamp("  POST /generate/finetuned - Generate with fine-tuned model") 
    log_with_timestamp("  POST /generate/both   - Generate with both models")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()