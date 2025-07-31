#!/usr/bin/env python3
"""
Exact Llama 3.2 11B VLM Fine-tuning Script
Replicates the exact working notebook implementation with TRL 0.15.2
Updated with STANDARD MLflow integration using MLflowCallback (like the provided example)
"""

import os
import sys
import warnings
import copy
import torch
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_environment():
    """Setup environment variables and directories"""
    # Environment setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HOME"] = "/mnt/jail/mnt/models/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/jail/mnt/models/cache/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/mnt/jail/mnt/datasets/cache"

    # Multi-GPU training environment
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    
    # Create necessary directories
    directories = [
        "/mnt/jail/mnt/models/logs",
        "/mnt/jail/mnt/models/cache",
        "/mnt/jail/mnt/models/fine-tuned",
        "/mnt/jail/mnt/jail/mnt/datasets/cache",
        "/mnt/jail/mnt/models/cache/huggingface",
        "/mnt/jail/mnt/models/cache/transformers"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Environment setup completed")

def setup_mlflow():
    """Setup MLflow tracking using standard MLflow client (like the provided example)"""
    try:
        # Check if MLflow environment variables are set
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        base_experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "vlm-finetuning")
        
        if not tracking_uri:
            print("Warning: MLFLOW_TRACKING_URI not set. Skipping MLflow setup.")
            return False, None
        
        if not username or not password:
            print("Warning: MLflow credentials not set. Skipping MLflow setup.")
            return False, None
        
        print(f"Setting up MLflow with tracking URI: {tracking_uri}")
        
        # Import MLflow
        import mlflow
        
        # Clear any certificate path that might cause issues
        if "MLFLOW_TRACKING_SERVER_CERT_PATH" in os.environ:
            cert_path = os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"]
            if not os.path.exists(cert_path):
                print(f"⚠ Removing invalid certificate path: {cert_path}")
                del os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"]
        
        # Clear SSL-related environment variables that might interfere
        ssl_env_vars = ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"]
        for var in ssl_env_vars:
            if var in os.environ:
                print(f"⚠ Clearing SSL environment variable: {var}")
                del os.environ[var]
        
        # Set up MLflow client (standard approach)
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        
        # Add timestamp to make experiment name unique
        import time
        timestamp = int(time.time())
        experiment_name = f"{base_experiment_name}-{timestamp}"
        
        print(f"Experiment name (with timestamp): {experiment_name}")
        print("Using default SSL verification (no certificate needed)")
        
        try:
            # Set the experiment (will create if doesn't exist)
            mlflow.set_experiment(experiment_name)
            print(f"✓ MLflow experiment set: {experiment_name}")
            
            # Test basic connectivity
            try:
                # Try to start and end a test run to verify everything works
                with mlflow.start_run(run_name="connectivity-test") as run:
                    mlflow.log_param("test_param", "connectivity_verification")
                    mlflow.log_metric("test_metric", 1.0)
                    print(f"   ✓ Test run created successfully: {run.info.run_id}")
                
                print("✓ MLflow setup completed successfully")
                print(f"  - Tracking URI: {tracking_uri}")
                print(f"  - Experiment: {experiment_name}")
                
                return True, experiment_name
                
            except Exception as test_e:
                print(f"✗ MLflow connectivity test failed: {test_e}")
                return False, None
                
        except Exception as exp_e:
            print(f"✗ Failed to set MLflow experiment: {exp_e}")
            return False, None
            
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")
        print("Training will continue without MLflow tracking...")
        return False, None

def check_header(prompt_header_seqs, current_seq):
    """Check if any prompt header sequence is in current sequence"""
    for header_seq in prompt_header_seqs:
        if len(current_seq) >= len(header_seq):
            for i in range(len(current_seq) - len(header_seq) + 1):
                if current_seq[i:i+len(header_seq)] == header_seq:
                    return True
    return False

def replace_target(target_seq, labels):
    """Replace target sequence with -100 in labels"""
    if len(labels) < len(target_seq):
        return labels
    
    for i in range(len(labels) - len(target_seq) + 1):
        if labels[i:i+len(target_seq)] == target_seq:
            labels[i:i+len(target_seq)] = [-100] * len(target_seq)
    return labels

def format_data(sample):
    """Format data for Llama 3.2 VLM training - exact from working notebook"""
    system_message = "You are an expert product description writer for Amazon."
    
    # Use the EXACT prompt template from the notebook
    prompt_template = """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""
    
    formatted_prompt = prompt_template.format(
        product_name=sample["Product Name"], 
        category=sample["Category"]
    )
    description = sample["description"]
    
    formatted_prompt = (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>{formatted_prompt}<|image|><|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>{description}<|eot_id|><|end_of_text|>"
    )
    return formatted_prompt

def tokenize(texts, images, processor):
    """Tokenize texts and images - exact from working notebook"""
    # texts is formatted data from samples: texts=format_data(samples)
    # tokenizer automatically appends '<|begin_of_text|>', so we replace it by empty string
    text_prompt = [prompt.replace('<|begin_of_text|>', '') for prompt in texts]
    batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    
    label_list = []
    for i in range(len(batch['input_ids'])):
        dialog_tokens = batch["input_ids"][i].tolist()  # i-th sequence
        labels = copy.copy(dialog_tokens)  # create a copy to avoid modifying dialog_tokens
        # get all indices of eot_id
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0

        # system prompt header "<|start_header_id|>system<|end_header_id|>" is tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" is tokenized to [128006, 882, 128007]
        # don't mask assistant header "<|start_header_id|>assistant<|end_header_id|>"
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for _, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]  # current_seq = last_idx to eot_idx
            # check if any prompt header is in current_seq
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header -> this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        
        # Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|> = [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)

        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:
                labels[i] = -100  # 128256 is image token index

        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch

class DataCollator:
    """Custom data collator - exact from working notebook"""
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, samples):
        # sample = [text,image] 
        formatted_texts, images = [], []
        for sample in samples:
            # extract formatted text and images
            image = sample["image"]  # PIL image object
            image = image.convert("RGB")
            images.append(image)
            
            # append text and image
            formatted_text = format_data(sample)
            formatted_texts.append(formatted_text)
        return tokenize(formatted_texts, images, self.processor)

def get_custom_dataset(split_ratio=0.95):
    """Load and split dataset - exact from working notebook"""
    from datasets import load_dataset
    
    dataset_dict = load_dataset(
        "philschmid/amazon-product-descriptions-vlm", 
        split="train", 
        cache_dir="/mnt/jail/mnt/datasets/cache"
    )
    # 95% train + 5% test
    dataset_dict = dataset_dict.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)
    # get train and test set
    ds_train, ds_test = dataset_dict['train'], dataset_dict['test']
    return ds_train, ds_test

def main():
    """Main training function - exact from working notebook with standard MLflow integration"""
    print("=== Llama 3.2 11B VLM Training Started (Standard MLflow Integration) ===")
    
    # Setup environment
    setup_environment()
    
    # Setup MLflow tracking - Using standard MLflow client
    mlflow_enabled, experiment_name = setup_mlflow()
    
    # Import required packages - exact versions from working notebook
    print("Importing required packages...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        from datasets import load_dataset
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        from peft import LoraConfig
        import peft
        print(f"PEFT version: {peft.__version__}")
        
        from trl import SFTConfig, SFTTrainer
        import trl
        print(f"TRL version: {trl.__version__}")
        
        # Import Hugging Face Hub for authentication
        from huggingface_hub import login, HfApi
        
        # Import MLflow components
        if mlflow_enabled:
            import mlflow
            from transformers.integrations import MLflowCallback
            print("✓ MLflow imports successful")
        
        print("All imports successful!")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install the exact versions from the working notebook:")
        print("pip install transformers==4.49.0 trl==0.15.2 datasets peft bitsandbytes accelerate pillow")
        sys.exit(1)

    # Handle Hugging Face authentication
    print("Setting up Hugging Face authentication...")
    try:
        # Try to authenticate with HF_TOKEN environment variable first
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            print("Found HF_TOKEN environment variable, logging in...")
            login(token=hf_token)
            print("✓ Authenticated with Hugging Face using environment token")
        else:
            print("No HF_TOKEN environment variable found.")
            print("Please set your Hugging Face token as an environment variable:")
            print("export HF_TOKEN=your_token_here")
            print("Or run: huggingface-cli login")
            
            # Try to use cached token
            try:
                api = HfApi()
                user_info = api.whoami()
                print(f"✓ Using cached authentication for user: {user_info['name']}")
            except Exception as auth_e:
                print(f"✗ Authentication failed: {auth_e}")
                print("Please authenticate with Hugging Face:")
                print("1. Run: huggingface-cli login")
                print("2. Or set HF_TOKEN environment variable")
                print("3. Make sure you have access to meta-llama/Llama-3.2-11B-Vision-Instruct")
                sys.exit(1)
                
    except Exception as e:
        print(f"Authentication setup error: {e}")
        print("Continuing without explicit authentication (using cached credentials)...")

    # Configuration - exact from working notebook
    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision"
    
    try:
        print("Loading dataset...")
        ds_train, ds_test = get_custom_dataset()
        print(f"Train dataset size: {len(ds_train)}")
        print(f"Test dataset size: {len(ds_test)}")
        print("Dataset loaded successfully")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print("Setting up Llama 3.2 11B VLM model and processor...")
    
    # Quantization config for memory efficiency - exact from working notebook
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # store weights in 4-bit
        bnb_4bit_use_double_quant=True,  # quantize the quantization
        bnb_4bit_quant_type="nf4",  # special quantization type to preserve statistical properties of weights
        # weights are stored in nf4. They are dequantized to bfloat16 for computation, output is in bfloat16
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model - exact from working notebook
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            token=os.environ.get('HF_TOKEN')
        )
        print("Model loaded successfully")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load processor - exact from working notebook
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=os.environ.get('HF_TOKEN')
        )
        print("Processor loaded successfully")
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    # LoRA configuration - exact from working notebook
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    print("LoRA configuration created")

    # Training arguments - exact from working notebook
    args = SFTConfig(
        output_dir="/mnt/jail/mnt/models/fine-tuned",  # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=2,          # batch size per device during training
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # Uses bfloat16 (16-bit)
        tf32=True,                              # Uses TF32 internally for matrix multiplications
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub
        report_to="mlflow" if mlflow_enabled else "none",  # Enable MLflow reporting
        gradient_checkpointing_kwargs={"use_reentrant": False}, # use reentrant checkpointing
        dataset_text_field="", # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True} # important for collator
    )
    args.remove_unused_columns = False
    
    # IMPORTANT: Set the experiment name in the training args to match our timestamped experiment
    if mlflow_enabled and experiment_name:
        # Override the default experiment name with our timestamped one
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        print(f"✓ Set MLFLOW_EXPERIMENT_NAME to: {experiment_name}")
    
    print("Training configuration created")

    # Setup MLflow callback if enabled
    callbacks = []
    mlflow_callback = None
    if mlflow_enabled:
        mlflow_callback = MLflowCallback()
        callbacks.append(mlflow_callback)
        print("✓ MLflow callback added to trainer")

    # Create trainer - exact from working notebook
    try:
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            data_collator=DataCollator(processor),
            peft_config=peft_config,
            processing_class=processor,  # Use processing_class (newer TRL)
            callbacks=callbacks,  # Add MLflow callback
        )
        print("SFTTrainer created successfully")
    except Exception as e:
        print(f"Error creating SFTTrainer with processing_class: {e}")
        # Fallback to tokenizer for older TRL versions
        try:
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=ds_train,
                data_collator=DataCollator(processor),
                peft_config=peft_config,
                tokenizer=processor.tokenizer,
                callbacks=callbacks,  # Add MLflow callback
            )
            print("SFTTrainer created successfully with tokenizer fallback")
        except Exception as e2:
            print(f"Error creating SFTTrainer with tokenizer: {e2}")
            sys.exit(1)

    # Setup MLflow run manually (like the provided example)
    run_id = None
    if mlflow_enabled and trainer.accelerator.is_main_process:
        try:
            import mlflow
            
            # Start a manual MLflow run
            mlflow_run = mlflow.start_run(run_name=f"llama-vlm-training-{int(time.time())}")
            run_id = mlflow_run.info.run_id
            print(f"✓ Manual MLflow run started with ID: {run_id}")
            
            # Log initial parameters
            mlflow.log_param("model_id", MODEL_ID)
            mlflow.log_param("train_dataset_size", len(ds_train))
            mlflow.log_param("test_dataset_size", len(ds_test))
            mlflow.log_param("dataset_name", "philschmid/amazon-product-descriptions-vlm")
            mlflow.log_param("quantization_load_in_4bit", True)
            mlflow.log_param("lora_alpha", 16)
            mlflow.log_param("lora_r", 8)
            mlflow.log_param("learning_rate", 2e-4)
            mlflow.log_param("num_train_epochs", 3)
            mlflow.log_param("per_device_train_batch_size", 1)
            mlflow.log_param("gradient_accumulation_steps", 16)
            mlflow.log_param("cuda_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            mlflow.log_param("experiment_name", experiment_name)
            print("✓ Initial parameters logged to MLflow")
            
        except Exception as mlflow_e:
            print(f"⚠ Manual MLflow setup failed: {mlflow_e}")
            print("Training will continue without MLflow logging")
            mlflow_enabled = False
            run_id = None

    print("Starting Llama 3.2 11B VLM training...")
    try:
        # Log training start time
        import time
        training_start_time = time.time()
        
        # Start training
        train_result = trainer.train()
        print("Training completed successfully!")
        
        # Post-training MLflow logging (like the provided example)
        if mlflow_enabled and trainer.accelerator.is_main_process and run_id:
            try:
                import mlflow
                # Log training results to the existing run
                training_end_time = time.time()
                training_duration = training_end_time - training_start_time
                
                mlflow.log_param("training_duration_minutes", training_duration / 60)
                mlflow.log_param("train_samples", len(ds_train))
                
                # Log final metrics from training
                if hasattr(train_result, 'metrics'):
                    for key, value in train_result.metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                
                print("✓ Final training metrics logged to MLflow")
                
                # End the MLflow run
                mlflow.end_run()
                print("✓ MLflow run ended successfully")
                
            except Exception as log_e:
                print(f"⚠ Error logging final metrics: {log_e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        # End MLflow run if it was started
        if mlflow_enabled and run_id:
            try:
                import mlflow
                mlflow.end_run(status="FAILED")
            except:
                pass
        sys.exit(1)

    print("Saving final model...")
    try:
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        print("Model saved successfully!")
        
        # Log model save status to existing MLflow run if still active
        if mlflow_enabled and trainer.accelerator.is_main_process and run_id:
            try:
                import mlflow
                # Check if run is still active
                active_run = mlflow.active_run()
                if active_run and active_run.info.run_id == run_id:
                    mlflow.log_param("model_saved_to", args.output_dir)
                    mlflow.log_param("model_save_status", "success")
                    print("✓ Model save status logged to MLflow")
            except Exception as save_log_e:
                print(f"⚠ Error logging model save status: {save_log_e}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)

    print("=== Llama 3.2 11B VLM Training completed successfully! ===")
    print(f"Model saved to: {args.output_dir}")
    if mlflow_enabled:
        print(f"Training metrics logged to MLflow experiment: {experiment_name}")
        if run_id:
            print(f"MLflow run ID: {run_id}")

if __name__ == "__main__":
    main()