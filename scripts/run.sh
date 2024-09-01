source venv/bin/activate
mkdir -p outputs/full

# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/train-10000" --output_dir "outputs/full/train-10000" --resume "detr-r50_no-class-head.pth"  --num_classes 4 --epochs 100

# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/dust-10/train" --output_dir "outputs/full/dust-10" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 & 
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/dust-100/train" --output_dir "outputs/full/dust-100" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/dust-1000/train" --output_dir "outputs/full/dust-1000" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &

# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/fog-10/train" --output_dir "outputs/full/fog-10" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/fog-100/train" --output_dir "outputs/full/fog-100" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/fog-1000/train" --output_dir "outputs/full/fog-1000" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &

# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/maple_leaf-10/train" --output_dir "outputs/full/maple_leaf-10" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 & 
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/maple_leaf-100/train" --output_dir "outputs/full/maple_leaf-100" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/maple_leaf-1000/train" --output_dir "outputs/full/maple_leaf-1000" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &

# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/rain-10/train" --output_dir "outputs/full/rain-10" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/rain-100/train" --output_dir "outputs/full/rain-100" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/rain-1000/train" --output_dir "outputs/full/rain-1000" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &

# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/snow-10/train" --output_dir "outputs/full/snow-10" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/snow-100/train" --output_dir "outputs/full/snow-100" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./aerial/snow-1000/train" --output_dir "outputs/full/snow-1000" --resume "outputs/full/train-10000/checkpoint.pth"  --num_classes 4 --epochs 200 &

python3 src/eval.py
python3 src/train_peft_model.py
