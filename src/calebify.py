import shutil
import os
import json

def calebify(name, prefix):
    input_annotations_path = f"aerial/{name}/train/annotations/custom_{prefix}.json"
    input_images_path = f"aerial/{name}/train/{prefix}2017"

    output_path = f"caleb/{name}/{prefix}"
    output_annotations_path = f"{output_path}/metadata.jsonl"
    output_images_path = f"{output_path}"

    if os.path.exists(output_path): shutil.rmtree(output_path)
    shutil.copytree(input_images_path, output_images_path, dirs_exist_ok=True)

    annotations = json.load(open(input_annotations_path, 'r'))
    annotation_images = annotations['images']
    annotation_annotations = annotations['annotations']

    metadata = {}
    for image in annotation_images:
        metadata[image["id"]] = {}
        metadata[image["id"]]["image_id"] = image["id"]
        metadata[image["id"]]["file_name"] = image["file_name"]
        metadata[image["id"]]["width"] = image["width"]
        metadata[image["id"]]["height"] = image["height"]
        metadata[image["id"]]["objects"] = {
            "id": [],
            "area": [],
            "bbox": [],
            "category": [],
        }

    for annotation in annotation_annotations:
        metadata[annotation["image_id"]]["objects"]["id"].append(annotation["id"])
        metadata[annotation["image_id"]]["objects"]["area"].append(annotation["area"])
        metadata[annotation["image_id"]]["objects"]["bbox"].append(annotation["bbox"])
        metadata[annotation["image_id"]]["objects"]["category"].append(annotation["category_id"])

    with open(output_annotations_path, 'w') as f:
        for key in metadata.keys():
            f.write(json.dumps(metadata[key]))
            f.write("\n")

def donkify(name):
    calebify(name, "train")
    calebify(name, "val")

try: donkify("train-10000")
except: pass
try: donkify("dust-10")
except: pass
try: donkify("dust-100")
except: pass
try: donkify("dust-1000")
except: pass
try: donkify("fog-10")
except: pass
try: donkify("fog-100")
except: pass
try: donkify("fog-1000")
except: pass
try: donkify("rain-10")
except: pass
try: donkify("rain-100")
except: pass
try: donkify("rain-1000")
except: pass
try: donkify("snow-10")
except: pass
try: donkify("snow-100")
except: pass
try: donkify("snow-1000")
except: pass
try: donkify("maple_leaf-10")
except: pass
try: donkify("maple_leaf-100")
except: pass
try: donkify("maple_leaf-1000")
except: pass
