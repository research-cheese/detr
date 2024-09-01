import json
import os
import shutil

from config.params import ID_2_LABEL

from utils.rm_makedir import rm_makedir
from PIL import Image, ImageChops

def extract_bbox(input_path, output_path, threshold):
    if os.path.exists(output_path): shutil.rmtree(output_path)

    with open(input_path, 'r') as f:
        data = json.load(f)

    probas = data[str(threshold)]['probas']
    bboxes = data[str(threshold)]['bboxes']
    
    open(output_path, 'w').write("")

    for proba, bbox in zip(probas, bboxes):
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        # Get argmax
        label_id = proba.index(max(proba))
        category_name = ID_2_LABEL[label_id]

        with open(output_path, 'a') as f:
            f.write(json.dumps({
                "category": category_name,
                "category_id": label_id,
                "confidence": max(proba),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "proba": proba
            }))
            f.write("\n")

def extract_ground_truth_bbox(annotation_path, output_path, image_name):
    annotations = json.load(open(annotation_path, 'r'))
    images = annotations['images']

    image_id = None
    for image in images:
        if image['file_name'] == image_name:
            image_id = image['id']

    bboxes = []
    for a in annotations['annotations']:
        if a['image_id'] == image_id:
            bbox = a['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]

            category = ID_2_LABEL[a['category_id']]

            bboxes.append({
                "category": category,
                "category_id": a['category_id'],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })
    
    if os.path.exists(output_path): os.rmdir(output_path)
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            f.write(json.dumps(bbox))
            f.write("\n")


def mariofy(name, threshold):
    output_path = f"mario/{threshold}/{name}"
    input_image_folder_path = f"aerial/test/val2017"
    metadata_folder_path = f"aerial/test/val2017"

    metadata_dict = {}
    for line in open(f"cityenviron/aerial/test/metadata.jsonl", 'r'):
        metadata = eval(line)
        metadata_dict[metadata["sample_index"]] = metadata


    rm_makedir(output_path)

    for image in os.listdir(input_image_folder_path):
        output_sample_path = f"{output_path}/{image}"
        os.makedirs(output_sample_path)

        input_image_path = f"{input_image_folder_path}/{image}"
        input_json_path = f"predictions/{name}/{image}.json"

        output_prediction_path = f"{output_sample_path}/prediction.jsonl"
        extract_bbox(input_json_path, output_prediction_path, threshold)

        shutil.copy(input_image_path, f"{output_sample_path}/image.png")

        output_bbox_path = f"{output_sample_path}/ground_truth.jsonl"
        extract_ground_truth_bbox(f"aerial/test/annotations/custom_val.json", output_bbox_path, image)

        output_metadata_path = f"{output_sample_path}/metadata.jsonl"
        input_image = Image.open(input_image_path).convert('RGB')
        for index in metadata_dict.keys():
            metadata_image = Image.open(f"cityenviron/aerial/test/images/{index}/Scene.png").convert("RGB")
            if ImageChops.difference(input_image, metadata_image).getbbox():
                metadata = metadata_dict[index]
                with open(output_metadata_path, 'w') as f:
                    f.write(json.dumps(metadata))
                break
        if os.path.exists(output_metadata_path) == False:
            print("Metadata not found for", image)



for threshold in [0.2, 0.4, 0.6, 0.8]:
    mariofy("train", threshold)
    mariofy("dust-0.5", threshold)
    mariofy("fog-0.5", threshold)
    mariofy("normal", threshold)
    mariofy("maple_leaf-0.5", threshold)
    mariofy("snow-0.5", threshold)
    mariofy("rain-0.5", threshold)
