python3 -m venv venv
source venv/bin/activate

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
pip install transformers
pip install datasets
pip install accelerate
pip install torchmetrics
pip install albumentations
pip install dataclasses
pip install peft
pip install timm

python3 src/download_checkpoint.py

# =============================================================================
# Create output directories
# =============================================================================
mkdir -p outputs/dust-0.5
mkdir -p outputs/maple_leaf-0.5
mkdir -p outputs/fog-0.5
mkdir -p outputs/snow-0.5
mkdir -p outputs/rain-0.5