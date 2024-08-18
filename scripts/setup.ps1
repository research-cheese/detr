python -m venv venv
./venv/Scripts/Activate

# =============================================================================
# Install the repository for DETR.
# =============================================================================
git clone https://github.com/woctezuma/detr.git
cd detr
git checkout finetune
cd ..

# =============================================================================
# Install the requirements for DETR.
# =============================================================================
pip install torchvision
pip install pycocotools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scipy

python src/download_checkpoint.py

# =============================================================================
# Create output directories
# =============================================================================
mkdir -p outputs/dust-0.5
mkdir -p outputs/maple_leaf-0.5
mkdir -p outputs/fog-0.5
mkdir -p outputs/snow-0.5
mkdir -p outputs/rain-0.5