#!/bin/bash
#SBATCH --job-name=vlm-exact-training
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8  # Using all of GPU's
#SBATCH --cpus-per-task=128 #Use all of 128 CPU's
#SBATCH --mem=1500G # Use most of the memory
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/models/logs/vlm-training_%j.out
#SBATCH --error=/mnt/models/logs/vlm-training_%j.err

echo "=== Llama 3.2 11B VLM Training Job Started (Fixed MLflow) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Initial working directory: $(pwd)"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "================================"

# Fix working directory issue - FULLY DYNAMIC
echo "=== Setting up Working Directory (Dynamic Detection) ==="

# Function to find the training script dynamically
find_training_script() {
    local script_name="train_llama32_vlm_mlflow.py"
    local search_paths=()
    
    echo "Searching for $script_name..."
    
    # Method 1: Use SLURM_SUBMIT_DIR (most reliable)
    if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
        echo "Checking SLURM submit directory: $SLURM_SUBMIT_DIR"
        if [ -f "$SLURM_SUBMIT_DIR/$script_name" ]; then
            echo "✓ Found script in SLURM submit directory"
            cd "$SLURM_SUBMIT_DIR" || return 1
            return 0
        fi
    fi
    
    # Method 2: Check current directory first
    echo "Checking current directory: $(pwd)"
    if [ -f "$script_name" ]; then
        echo "✓ Found script in current directory"
        return 0
    fi
    
    # Method 3: Search filesystem dynamically
    echo "Performing filesystem search for $script_name..."
    
    # Build dynamic search paths based on what actually exists
    local potential_roots=("/mnt" "/home" "/opt" "/usr/local" ".")
    
    for root in "${potential_roots[@]}"; do
        if [ -d "$root" ]; then
            echo "  Searching in $root..."
            # Find with timeout to avoid hanging on large filesystems
            timeout 30 find "$root" -name "$script_name" -type f -readable 2>/dev/null | while read -r found_script; do
                script_dir=$(dirname "$found_script")
                echo "  Found: $found_script"
                search_paths+=("$script_dir")
            done
        fi
    done
    
    # Method 4: Use find command with common directories
    echo "  Searching common project directories..."
    local common_patterns=(
        "*training*"
        "*vlm*"
        "*llama*"
        "*models*"
        "*finetun*"
        "*fine-tun*"
    )
    
    for pattern in "${common_patterns[@]}"; do
        while IFS= read -r -d '' dir; do
            if [ -f "$dir/$script_name" ]; then
                echo "  Found in pattern-matched directory: $dir"
                search_paths+=("$dir")
            fi
        done < <(find /mnt -type d -name "$pattern" -print0 2>/dev/null | head -20)
    done
    
    # Method 5: Direct filesystem search (last resort)
    echo "  Performing direct filesystem search..."
    local script_locations
    readarray -t script_locations < <(find /mnt -name "$script_name" -type f 2>/dev/null | head -10)
    
    for script_path in "${script_locations[@]}"; do
        if [ -f "$script_path" ]; then
            script_dir=$(dirname "$script_path")
            echo "  Found via direct search: $script_path"
            search_paths+=("$script_dir")
        fi
    done
    
    # Try each found location
    for script_dir in "${search_paths[@]}"; do
        if [ -d "$script_dir" ] && [ -f "$script_dir/$script_name" ]; then
            echo "✓ Attempting to use: $script_dir"
            if cd "$script_dir" 2>/dev/null; then
                echo "✓ Successfully changed to: $(pwd)"
                return 0
            else
                echo "✗ Cannot access directory: $script_dir"
            fi
        fi
    done
    
    return 1
}

# Function to validate the environment
validate_environment() {
    local script_name="train_llama32_vlm_mlflow.py"
    
    echo "=== Environment Validation ==="
    echo "Current directory: $(pwd)"
    echo "Script exists: $(test -f "$script_name" && echo 'YES' || echo 'NO')"
    
    if [ ! -f "$script_name" ]; then
        echo "ERROR: $script_name not found in current directory"
        echo ""
        echo "Directory contents:"
        ls -la
        echo ""
        echo "Python files in current directory:"
        find . -maxdepth 1 -name "*.py" -type f 2>/dev/null || echo "  No Python files found"
        echo ""
        echo "Searching for any Python files nearby..."
        find . -maxdepth 2 -name "*.py" -type f 2>/dev/null | head -10
        return 1
    fi
    
    # Check if script is readable and not empty
    if [ ! -r "$script_name" ]; then
        echo "ERROR: $script_name is not readable"
        return 1
    fi
    
    if [ ! -s "$script_name" ]; then
        echo "ERROR: $script_name is empty"
        return 1
    fi
    
    echo "✓ Script validation passed"
    return 0
}

# Execute the dynamic search
if find_training_script; then
    echo "✓ Successfully located training script"
    if validate_environment; then
        echo "✓ Environment validation passed"
    else
        echo "✗ Environment validation failed"
        exit 1
    fi
else
    echo "✗ Failed to locate training script"
    echo ""
    echo "=== Troubleshooting Information ==="
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-'Not set'}"
    echo "Current directory: $(pwd)"
    echo "User: $(whoami)"
    echo "Home directory: $HOME"
    echo ""
    echo "=== Manual Search Results ==="
    echo "Looking for train_llama32_vlm_mlflow.py in filesystem..."
    find /mnt /home -name "train_llama32_vlm_mlflow.py" -type f 2>/dev/null | head -10 || echo "No instances found"
    echo ""
    echo "Looking for any Python training scripts..."
    find /mnt -name "*train*.py" -type f 2>/dev/null | head -10 || echo "No training scripts found"
    exit 1
fi

echo "================================"

# MLflow Configuration - NO CERTIFICATE NEEDED
echo "=== Setting up MLflow Configuration (No Certificate) ==="

# Check if required MLflow variables are set
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    echo "✗ MLFLOW_TRACKING_URI not set"
    echo "Please set: export MLFLOW_TRACKING_URI=your_mlflow_server_url"
    exit 1
fi

if [ -z "$MLFLOW_TRACKING_USERNAME" ]; then
    echo "✗ MLFLOW_TRACKING_USERNAME not set"
    echo "Please set: export MLFLOW_TRACKING_USERNAME=your_username"
    exit 1
fi

if [ -z "$MLFLOW_TRACKING_PASSWORD" ]; then
    echo "✗ MLFLOW_TRACKING_PASSWORD not set"
    echo "Please set: export MLFLOW_TRACKING_PASSWORD=your_password"
    exit 1
fi

# Optional variables with defaults
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-vlm-finetuning}"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="${MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING:-true}"

# IMPORTANT: Remove any certificate path that might cause issues
if [ -n "$MLFLOW_TRACKING_SERVER_CERT_PATH" ]; then
    echo "⚠ Removing MLFLOW_TRACKING_SERVER_CERT_PATH (not needed)"
    unset MLFLOW_TRACKING_SERVER_CERT_PATH
fi

# Clear SSL environment variables that might interfere
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE  
unset SSL_CERT_FILE

echo "✓ Required MLflow environment variables are set"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow experiment name: $MLFLOW_EXPERIMENT_NAME"
echo "ℹ Using default SSL verification (no certificate - same as test_mlflow.py)"

# Test MLflow connectivity using Python (NO CERTIFICATE)
echo "=== Testing MLflow Connectivity (No Certificate) ==="

# Ensure pip and requests are available
echo "Ensuring requests library is available..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: No Python interpreter found"
    exit 1
fi

if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
elif $PYTHON_CMD -c "import pip" &> /dev/null; then
    PIP_CMD="$PYTHON_CMD -m pip"
else
    echo "Installing pip..."
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user --no-warn-script-location
    rm -f get-pip.py
    export PATH="$HOME/.local/bin:$PATH"
    PIP_CMD="$PYTHON_CMD -m pip"
fi

$PIP_CMD install requests --user --root-user-action=ignore 2>/dev/null || echo "Requests already installed"

# Test MLflow connectivity using Python requests (like test_mlflow.py)
$PYTHON_CMD -c "
import os
import sys

def test_mlflow_connectivity():
    try:
        import requests
        from requests.auth import HTTPBasicAuth
        import json
        
        # Get configuration from environment
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        username = os.environ.get('MLFLOW_TRACKING_USERNAME')
        password = os.environ.get('MLFLOW_TRACKING_PASSWORD')
        cert_path = os.environ.get('MLFLOW_TRACKING_SERVER_CERT_PATH')
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'vlm-finetuning')
        
        if not all([tracking_uri, username, password]):
            print('✗ MLflow environment variables not properly set')
            return False
        
        print(f'Testing MLflow connectivity to: {tracking_uri}')
        print('Using default SSL verification (no certificate)')
        
        # Setup authentication (no SSL certificate)
        auth = HTTPBasicAuth(username, password)
        headers = {'Content-Type': 'application/json'}
        
        # Test basic connectivity (no certificate)
        print('1. Testing basic connectivity...')
        response = requests.get(
            f'{tracking_uri}/api/2.0/mlflow/experiments/search?max_results=10',
            auth=auth,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f'   ✗ Failed to connect: HTTP {response.status_code}')
            print(f'   Response: {response.text[:200]}...')
            return False
        
        print('   ✓ Basic connectivity successful')
        
        # Test experiment access
        print(f'2. Testing experiment access for: {experiment_name}')
        experiments = response.json().get(\"experiments\", [])
        existing_exp = None
        
        for exp in experiments:
            if exp.get(\"name\") == experiment_name:
                existing_exp = exp
                break
        
        if existing_exp:
            print(f'   ✓ Found existing experiment: {experiment_name} (ID: {existing_exp[\"experiment_id\"]})')
        else:
            print(f'   ℹ Experiment {experiment_name} not found - will be created during training')
        
        print('✓ MLflow connectivity test passed - ready for training!')
        return True
        
    except requests.exceptions.SSLError as ssl_e:
        print(f'✗ SSL Error: {ssl_e}')
        print('Certificate path issues detected. Check:')
        print(f'  1. Certificate file exists: {cert_path}')
        print('  2. Certificate file is valid')
        print('  3. Certificate matches the server')
        return False
        
    except requests.exceptions.ConnectionError as conn_e:
        print(f'✗ Connection Error: {conn_e}')
        print('Check if MLflow server is accessible from this network')
        return False
        
    except requests.exceptions.Timeout as timeout_e:
        print(f'✗ Timeout Error: {timeout_e}')
        print('MLflow server is not responding within the timeout period')
        return False
        
    except Exception as e:
        print(f'✗ Unexpected error: {e}')
        return False

# Run the test
if test_mlflow_connectivity():
    print()
    print('🎉 MLflow is ready for training!')
    sys.exit(0)
else:
    print()
    print('⚠ MLflow connectivity issues detected')
    print('Training will continue but MLflow logging may fail')
    sys.exit(0)  # Don't fail the job, just warn
"

echo "================================"

# Add better error handling
set -o pipefail

# Create necessary directories dynamically
echo "=== Creating Necessary Directories ==="
directories=(
    "/mnt/models/logs"
    "/mnt/models/cache"
    "/mnt/models/fine-tuned"
    "/mnt/datasets/cache"
    "/mnt/models/cache/huggingface"
    "/mnt/models/cache/transformers"
)

for directory in "${directories[@]}"; do
    if mkdir -p "$directory" 2>/dev/null; then
        echo "✓ Created/verified: $directory"
    else
        echo "✗ Failed to create: $directory"
    fi
done

# Check GPU availability
echo "=== GPU Information ==="
nvidia-smi
echo "======================="

# Set up Python environment
echo "=== Setting up Python Environment ==="

# Update PATH to include user-installed packages
export PATH="$HOME/.local/bin:$PATH"

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD --version

echo "Using pip command: $PIP_CMD"
$PIP_CMD --version

# Clean up any existing installations to avoid conflicts
echo "Cleaning up existing installations..."
$PIP_CMD uninstall -y transformers trl peft bitsandbytes accelerate 2>/dev/null || echo "No existing packages to uninstall"

# Install exact versions from the working notebook
echo "Installing exact package versions from working notebook..."
$PIP_CMD install --upgrade pip --user --root-user-action=ignore

# Install PyTorch first (compatible with CUDA 12.1)
echo "Installing PyTorch with CUDA 12.1 support..."
$PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --user --root-user-action=ignore

# Install exact versions that worked in the notebook
echo "Installing transformers==4.49.0..."
$PIP_CMD install transformers==4.49.0 --user --root-user-action=ignore

echo "Installing trl==0.15.2..."
$PIP_CMD install trl==0.15.2 --user --root-user-action=ignore

echo "Installing other required packages..."
$PIP_CMD install datasets peft bitsandbytes accelerate pillow requests --user --root-user-action=ignore

echo "Installing MLflow and system monitoring dependencies..."
$PIP_CMD install mlflow psutil pynvml --user --root-user-action=ignore

# Ensure user-installed packages are in PATH
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.*/site-packages:$PYTHONPATH"

echo "Package installation completed."

# Verify installations
echo "=== Verifying Package Versions ==="
$PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.executable}')
print()

try:
    import transformers
    import trl
    import peft
    import datasets
    print(f'✓ Transformers version: {transformers.__version__}')
    print(f'✓ TRL version: {trl.__version__}')
    print(f'✓ PEFT version: {peft.__version__}')
    print(f'✓ Datasets version: {datasets.__version__}')
except ImportError as e:
    print(f'✗ Failed to import basic packages: {e}')
    sys.exit(1)

# Test critical imports
try:
    from transformers import MllamaForConditionalGeneration, MllamaProcessor
    print('✓ MllamaForConditionalGeneration and MllamaProcessor available')
except ImportError as e:
    print('✗ Llama 3.2 Vision classes not available:', e)

try:
    from trl import SFTConfig, SFTTrainer
    print('✓ SFTConfig and SFTTrainer available')
except ImportError as e:
    print('✗ TRL classes not available:', e)

try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ CUDA version: {torch.version.cuda}')
        print(f'✓ GPU count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'✗ PyTorch not available: {e}')
"

# Handle Hugging Face authentication
echo "=== Setting up Hugging Face Authentication ==="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠ HF_TOKEN environment variable not set"
    echo "Please set: export HF_TOKEN=your_huggingface_token"
    echo "Trying to use cached authentication from huggingface-cli login..."
    
    # Try to check if user is already logged in
    $PYTHON_CMD -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    user_info = api.whoami()
    print(f'✓ Already authenticated as: {user_info[\"name\"]}')
except Exception as e:
    print(f'✗ Not authenticated: {e}')
    print('Please run: huggingface-cli login')
    print('Or set HF_TOKEN environment variable')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "Authentication failed. Exiting."
        exit 1
    fi
else
    echo "✓ HF_TOKEN environment variable found"
fi

echo "================================"

# Final verification before running training
echo "=== Final Pre-Training Verification ==="
echo "Current directory: $(pwd)"
echo "Python script exists: $(test -f train_llama32_vlm_mlflow.py && echo 'YES' || echo 'NO')"
echo "Python command: $PYTHON_CMD"

# Show final environment summary
echo ""
echo "=== Environment Summary ==="
echo "Working Directory: $(pwd)"
echo "Python Command: $PYTHON_CMD"
echo "HF Authentication: $([ -n "$HF_TOKEN" ] && echo 'Token Set' || echo 'Using cached login')"
echo "MLflow Certificate: Not needed (no certificate - using default SSL)"
echo "GPU Count: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'Unknown')"
echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}' || echo 'Unknown')"
echo "================================"

# Run the FIXED training script
echo "Starting FIXED training script with proper MLflow integration..."
$PYTHON_CMD train_llama32_vlm_mlflow.py

echo "Training job completed at $(date)"
echo "Check logs at: /mnt/models/logs/"
echo "Model saved to: /mnt/models/fine-tuned/"