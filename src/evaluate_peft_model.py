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
from peft.peft_model import PeftModel

def load_image(image_path):
    image = Image.open(image_path)
    return image


def eval_checkpoint(checkpoint="facebook/detr-resnet-50", dataset_name="val"):
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    transform = albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
        ignore_mismatched_sizes=True,
    )
    model.to("cuda")

    images_dir = f"caleb/{dataset_name}/train"
    for image_dir in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_dir)
        image = load_image(image_path)
        inputs = image_processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to("cuda"))
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
        print(results)
        input("WAIT!")


def eval_peft_model(
    checkpoint="facebook/detr-resnet-50",
    base_checkpoint="facebook/detr-resnet-50",
    dataset_name="val",
):

    model = AutoModelForObjectDetection.from_pretrained(
        base_checkpoint,
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(
        model,
        model_id=checkpoint,
    )

    images_dir = f"caleb/{dataset_name}"
    for image_dir in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_dir)
        image = load_image(image_path)
        outputs = model(image)
        print(outputs)
        input("WAIT!")


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

for dataset_name in ["val", "test"]:
    base_checkpoint_path = f"outputs/hugging-full/train-10000/checkpoint.pth"
    eval_checkpoint(checkpoint=base_checkpoint_path, dataset_name=dataset_name)
    for config in [
        ("IA3", ia3_config),
        ("LoRA", lora_config),
        ("LNTuning", lntuning_config),
    ]:
        for model_name in [
            "dust-10/train",
            "dust-100/train",
            "dust-1000/train",
            "fog-10/train",
            "fog-100/train",
            "fog-1000/train",
            "maple_leaf-10/train",
            "maple_leaf-100/train",
            "maple_leaf-1000/train",
            "rain-10/train",
            "rain-100/train",
            "rain-1000/train",
            "snow-10/train",
            "snow-100/train",
            "snow-1000/train",
        ]:
            checkpoint_path = f"outputs/{config[1]}/{model_name}/checkpoint.pth"
            eval_peft_model(
                checkpoint_path=checkpoint_path,
                base_checkpoint=base_checkpoint_path,
                dataset_name=dataset_name,
            )
