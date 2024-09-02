from transformers import AutoImageProcessor
from datasets import load_dataset
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer
from accelerate import Accelerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from transformers.image_transforms import center_to_corners_format
from functools import partial

import albumentations
import numpy as np
import torch
from dataclasses import dataclass


from config.params import LABEL_2_ID, ID_2_LABEL
import os
import shutil
from PIL import Image
from peft import IA3Config, LoraConfig, LNTuningConfig, get_peft_model


def load_pretrained(config, name, checkpoint="facebook/detr-resnet-50", prefix="lora"):
    origin = f"{name}"
    
    data = {
        "train": f"aerial/{origin}/annotations/custom_train.json",
        "validation": f"aerial/{origin}/annotations/custom_val.json",
    }
    cs_caronly = load_dataset("json", data_files=data)

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    transform = albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )
        
    def formatted_anns(image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations


    def transform_aug_ann_test(examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        examples["image"] = [
            Image.open(f"{origin}/train/{file[:-4]}.jpg") for file in examples["file_name"]
        ]
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = transform(
                image=image, bboxes=objects["bbox"], category=objects["category"]
            )

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")


    def transform_aug_ann_val(examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        examples["image"] = [
            Image.open(f"{origin}/val/{file[:-4]}.jpg") for file in examples["file_name"]
        ]
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = transform(
                image=image, bboxes=objects["bbox"], category=objects["category"]
            )

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")


    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch


    def convert_bbox_yolo_to_pascal(boxes, image_size):
        boxes = center_to_corners_format(boxes)

        height, width = image_size
        boxes = boxes * torch.tensor([[width, height, width, height]])
        return boxes


    @dataclass
    class ModelOutput:
        logits: torch.Tensor
        pred_boxes: torch.Tensor


    @torch.no_grad()
    def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = []
        post_processed_targets = []
        post_processed_predictions = []

        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
            for image_target in batch:
                boxes = torch.tensor(image_target["boxes"])
                boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
                labels = torch.tensor(image_target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})

        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)

        metric1 = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
        metric1.update(post_processed_predictions, post_processed_targets)
        metric1 = metric1.compute()
        metric1 = {k: round(v.item(), 4) for k, v in metric1.items()}

        metric2 = CompleteIntersectionOverUnion(box_format="xyxy", class_metrics=False)
        metric2.update(post_processed_predictions, post_processed_targets)
        metric2 = metric2.compute()
        metric2 = {k: round(v.item(), 4) for k, v in metric2.items()}

        metrics = dict(metric1, **metric2)
        return metrics

    cs_caronly["train"] = cs_caronly["train"].with_transform(transform_aug_ann_test)
    cs_caronly["validation"] = cs_caronly["validation"].with_transform(
        transform_aug_ann_val
    )

    eval_compute_metrics_fn = partial(
        compute_metrics,
        image_processor=image_processor,
        id2label=ID_2_LABEL,
        threshold=0.0,
    )

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
        ignore_mismatched_sizes=True,
    )
    model = get_peft_model(model, config)
    model.from_pretrained(f"outputs/{prefix}/{name}/model.pth")
    return model

def evaluate_model(config, name, checkpoint, prefix, test_images_folder, output_folder_path):
    # Load
    model = load_pretrained(config, name, checkpoint=checkpoint, prefix=prefix)

    output_folder = f"{output_folder_path}/{name}"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for image in os.listdir(test_images_folder):
        results = model(image)
        print(results)
        input("Bonkus?")
        print(results)            

ia3_config = IA3Config(
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
    feedforward_modules=["fc1", "fc2"],
    modules_to_save=["class_labels_classifier", "bbox_predictor"],
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
    modules_to_save=["class_labels_classifier", "bbox_predictor"],
)

lntuning_config = LNTuningConfig(
    target_modules=[
        "self_attn_layer_norm",
        "final_layer_norm",
        "encoder_attn_layer_norm",
        "layernorm",
    ],
    modules_to_save=["class_labels_classifier", "bbox_predictor"],
)

for config in [
    ("IA3", ia3_config),
    ("LoRA", lora_config),
    ("LNTuning", lntuning_config),
]:
    evaluate_model(config[1], "train-10000", prefix=config[0])
    evaluate_model(config[1], "dust-10", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "dust-100", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "dust-1000", "outputs/train-10000/model.pth", prefix=config[0])

    evaluate_model(config[1], "fog-10", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "fog-100", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "fog-1000", "outputs/train-10000/model.pth", prefix=config[0])
    
    evaluate_model(config[1], "maple_leaf-10", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "maple_leaf-100", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "maple_leaf-1000", "outputs/train-10000/model.pth", prefix=config[0])
    
    evaluate_model(config[1], "rain-10", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "rain-100", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "rain-1000", "outputs/train-10000/model.pth", prefix=config[0])
    
    evaluate_model(config[1], "snow-10", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "snow-100", "outputs/train-10000/model.pth", prefix=config[0])
    evaluate_model(config[1], "snow-1000", "outputs/train-10000/model.pth", prefix=config[0])
