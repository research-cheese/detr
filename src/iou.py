from PIL import Image, ImageChops
import json
import math
import os

from config.params import LABEL_2_ID
from sklearn.metrics import classification_report
import numpy as np

def calculate_iou(sample, ax=None):
    ground_truth_path = f"{sample}/ground_truth.jsonl"
    prediction_path = f"{sample}/prediction.jsonl"
    image_path = f"{sample}/image.png"

    image = Image.open(image_path)
    ret = {}
    for label in LABEL_2_ID.keys():
        ground_truth = [json.loads(line) for line in open(ground_truth_path, 'r')]
        predictions = [json.loads(line) for line in open(prediction_path, 'r')]                

        # Filter out only the current label
        ground_truth = [g for g in ground_truth if g['class'] == label]
        predictions = [p for p in predictions if p['class'] == label]

        # Calculate the intersection over union. Create binary masks for the ground truth and predictions
        ground_truth_mask = np.zeros(image.size, dtype=np.uint8)
        prediction_mask = np.zeros(image.size, dtype=np.uint8)

        for g in ground_truth:
            xmin = int(g['xmin'])
            ymin = int(g['ymin'])
            xmax = int(g['xmax'])
            ymax = int(g['ymax'])
            ground_truth_mask[ymin:ymax, xmin:xmax] = 1
        
        for p in predictions:
            xmin = int(p['xmin'])
            ymin = int(p['ymin'])
            xmax = int(p['xmax'])
            ymax = int(p['ymax'])
            prediction_mask[ymin:ymax, xmin:xmax] = 1

        intersection = np.bitwise_and(ground_truth_mask, prediction_mask)
        union = np.bitwise_or(ground_truth_mask, prediction_mask)
        iou = 0 if union.sum() == 0 else intersection.sum() / union.sum()
        
        report = classification_report(
            prediction_mask.flatten(), 
            ground_truth_mask.flatten(), 
            output_dict=True
        )

        ret[label] = {
            "iou": iou,
            "report": report
        }
    return ret

def evaluate_dataset(dataset):
    ious = []
    for sample in os.listdir(dataset):
        sample_path = f"{dataset}/{sample}"
        ious.append(calculate_iou(sample_path))
        
    ret = {}
    for label in LABEL_2_ID.keys():
        iou = sum([iou[label] for iou in ious]) / len(ious)
        ret[label] = iou
    return ret

for threshold in [0.4, 0.6, 0.8]:
    with open(f"thresholds/thresho-{threshold}.json", "w") as b:
        b.write(f"THRESHOLD {threshold}")
        b.write("DUST 0.25", evaluate_dataset(f"mario/{threshold}/dust-0.25"))
        b.write("FOG 0.25", evaluate_dataset(f"mario/{threshold}/fog-0.25"))
        b.write("MAPLE LEAF 0.5", evaluate_dataset(f"mario/{threshold}/maple_leaf-0.5"))
        b.write("NORMAL", evaluate_dataset(f"mario/{threshold}/normal"))
        b.write("RAIN 0.5", evaluate_dataset(f"mario/{threshold}/rain-0.5"))
        b.write("SNOW 0.5", evaluate_dataset(f"mario/{threshold}/snow-0.5"))
