import os
import json
import shutil

import torch
import torchvision.transforms as T

from PIL import Image

NUM_CLASSES = 4

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs, im,
                               threshold=0.7):

  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

  return probas_to_keep, bboxes_scaled

def load_model(name):
    # Load
    model = torch.hub.load('facebookresearch/detr',
                        'detr_resnet50',
                        pretrained=False,
                        num_classes=NUM_CLASSES)

    checkpoint = torch.load(f'outputs/{name}/checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                        strict=False)

    model.eval()

    return model

def run_worflow(my_image, my_model):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(my_image).unsqueeze(0)

  # propagate through the model
  outputs = my_model(img)
  
  ret = {}
  for threshold in [0.8, 0.6, 0.4, 0.2, 0.0]:

    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, my_image,
                                                              threshold=threshold)
    
    ret[threshold] = {
        'probas': probas_to_keep.tolist(),
        'bboxes': bboxes_scaled.tolist()
    }
  return ret

def evaluate_model(name):
    print(f"Processing {name}")

    # Load
    model = load_model(name)

    test_images_folder = "aerial/test/val2017"
    output_folder = f"predictions/{name}"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for image in os.listdir(test_images_folder):
        im = Image.open(f"{test_images_folder}/{image}")
        im = im.convert('RGB')
        ret = run_worflow(im, model)

        with open(f"{output_folder}/{image}.json", 'w') as f:
            json.dump(ret, f)

evaluate_model("train")
# evaluate_model("dust-0.5")
# evaluate_model("fog-0.5")
# evaluate_model("normal")
# evaluate_model("rain-0.5")
# evaluate_model("snow-0.5")
# evaluate_model("maple_leaf-0.5")