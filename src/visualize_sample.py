from PIL import Image
import json
import matplotlib.pyplot as plt

from config.params import LABEL_2_ID

def visualize_sample(sample, ax=None):
    print("VISUALIZING", sample)
    ground_truth_path = f"{sample}/ground_truth.jsonl"
    prediction_path = f"{sample}/prediction.jsonl"
    image_path = f"{sample}/image.png"

    image = Image.open(image_path)
    
    for label in LABEL_2_ID.keys():
        plt.imshow(image)
        print(label)
        with open(ground_truth_path, 'r') as f:
            ground_truth = [json.loads(line) for line in f]

            for g in ground_truth:
                xmin = g['xmin']
                ymin = g['ymin']
                xmax = g['xmax']
                ymax = g['ymax']
                category = g['class']

                if category != label: continue

                plt.plot([xmin, xmax, xmax, xmin, xmin],
                        [ymin, ymin, ymax, ymax, ymin], label=category, color='red')
                
        # with open(prediction_path, 'r') as f:
        #     predictions = [json.loads(line) for line in f]

        #     for p in predictions:
        #         xmin = p['xmin']
        #         ymin = p['ymin']
        #         xmax = p['xmax']
        #         ymax = p['ymax']
        #         category = p['class']

        #         if category != label: continue

        #         plt.plot([xmin, xmax, xmax, xmin, xmin],
        #                 [ymin, ymin, ymax, ymax, ymin], label=category, color='red')
                
        plt.show()

visualize_sample("mario/0.4/maple_leaf-0.5/val2017_1110.png")