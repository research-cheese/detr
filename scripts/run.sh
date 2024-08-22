source venv/bin/activate

python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/train" --output_dir "outputs/train" --resume "detr-r50_no-class-head.pth"  --num_classes 4 --epochs 40
python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/dust-0.5" --output_dir "outputs/dust-0.5" --resume "outputs/train/checkpoint.pth"  --num_classes 4 --epochs 40
python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/fog-0.5" --output_dir "outputs/fog-0.5" --resume "outputs/train/checkpoint.pth"  --num_classes 4 --epochs 40
python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/maple_leaf-0.5" --output_dir "outputs/maple_leaf-0.5" --resume "outputs/train/checkpoint.pth"  --num_classes 4 --epochs 40
python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/normal" --output_dir "outputs/normal" --resume "outputs/train/checkpoint.pth"  --num_classes 4 --epochs 40
python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/rain-0.5" --output_dir "outputs/rain-0.5" --resume "outputs/train/checkpoint.pth"  --num_classes 4 --epochs 40
python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/snow-0.5" --output_dir "outputs/snow-0.5" --resume "outputs/train/checkpoint.pth"  --num_classes 4 --epochs 40

python3 src/eval.py