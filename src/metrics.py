from typing import List

def iou (box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1, y1, x2, y2 = max(x1, x2), max(y1, y2), min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union

def calculate_tp (ground_truth_boxes, prediction_boxes, iou_threshold):
    tp = 0
    for g in ground_truth_boxes:
        for p in prediction_boxes:
            iou = iou(g, p)
            if iou > iou_threshold:
                tp += 1
                break
    return tp
