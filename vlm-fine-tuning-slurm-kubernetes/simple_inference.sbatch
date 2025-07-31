#!/bin/bash
#SBATCH --job-name=simple-inference
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/models/logs/simple_inference_%j.out
#SBATCH --error=/mnt/models/logs/simple_inference_%j.err

echo "=== Simple Inference Server Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "============================================"

# Set environment variables
export HF_TOKEN=${HF_TOKEN}
export HF_HOME="/mnt/jail/mnt/models/cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/jail/mnt/models/cache/transformers"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Create necessary directories
mkdir -p /mnt/models/logs
mkdir -p /mnt/jail/mnt/models/cache/huggingface
mkdir -p /mnt/jail/mnt/models/cache/transformers

# Create working directory in /tmp
WORK_DIR="/tmp/simple_inference_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

echo "=== Copying files to working directory ==="
echo "Working directory: $WORK_DIR"

# Copy files from the source directory (try multiple possible locations)
SOURCE_DIRS=(
    "/mnt/jail/mnt/training/vlm-fine-tuning-slurm-kubernetes"
    "/mnt/training/vlm-fine-tuning-slurm-kubernetes"
    "/training/vlm-fine-tuning-slurm-kubernetes"
)

FILES_COPIED=false
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    if [ -f "$SOURCE_DIR/simple_inference_server.py" ]; then
        echo "✓ Found files in: $SOURCE_DIR"
        cp "$SOURCE_DIR/simple_inference_server.py" "$WORK_DIR/"
        cp "$SOURCE_DIR/requirements.txt" "$WORK_DIR/" 2>/dev/null || echo "requirements.txt not found, will create"
        FILES_COPIED=true
        break
    fi
done

if [ "$FILES_COPIED" = false ]; then
    echo "ERROR: Could not find simple_inference_server.py in any of these locations:"
    for dir in "${SOURCE_DIRS[@]}"; do
        echo "  - $dir"
    done
    exit 1
fi

# Change to working directory
cd "$WORK_DIR"
echo "Changed to working directory: $(pwd)"

# Create simple requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "Creating simple requirements.txt..."
    cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.40.0
peft>=0.4.0
flask>=2.0.0
flask-cors>=4.0.0
pillow>=9.0.0
huggingface-hub>=0.16.0
requests>=2.25.0
EOF
fi

echo "Files in working directory:"
ls -la

# Setup Python environment
echo "=== Setting up Python Environment ==="
python3 --version
export PATH="$HOME/.local/bin:$PATH"

# Determine pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD="python3 -m pip"
fi

echo "Using pip command: $PIP_CMD"

# Install requirements
echo "Installing Python requirements..."
$PIP_CMD install --user -r requirements.txt --root-user-action=ignore

# Update PYTHONPATH
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$HOME/.local/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH"

# Set fine-tuned model path
echo "=== Setting Fine-tuned Model Path ==="
MODEL_PATH="/mnt/jail/mnt/jail/mnt/models/fine-tuned"

if [ -d "$MODEL_PATH" ]; then
    echo "✓ Found fine-tuned model at: $MODEL_PATH"
elif [ -d "/mnt/jail/mnt/models/fine-tuned" ]; then
    MODEL_PATH="/mnt/jail/mnt/models/fine-tuned"
    echo "✓ Found fine-tuned model at: $MODEL_PATH (fallback location)"
else
    echo "ERROR: Fine-tuned model not found at either location:"
    echo "  - /mnt/jail/mnt/jail/mnt/models/fine-tuned"
    echo "  - /mnt/jail/mnt/models/fine-tuned"
    exit 1
fi

export FINE_TUNED_MODEL_PATH="$MODEL_PATH"

# Verify packages
echo "=== Verifying Package Installation ==="
python3 -c "
import sys
print(f'Python version: {sys.version}')
try:
    import transformers, torch, flask, peft
    print(f'✓ Transformers: {transformers.__version__}')
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ GPU count: {torch.cuda.device_count()}')
    print('✓ All packages imported successfully')
except ImportError as e:
    print(f'✗ Package import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

# Start the server
echo "=== Starting Simple Inference Server ==="
echo "Server will be available on port 5000"
echo "Model path: $MODEL_PATH"
echo ""
echo "Starting at: $(date)"

python3 simple_inference_server.py

echo "Server completed at $(date)"

# Cleanup
echo "Cleaning up working directory..."
cd /tmp
rm -rf "$WORK_DIR"