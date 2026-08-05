# import
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
roboflow_folder = PROJECT_ROOT / "data" / "masks_roboflow"
mask_folder = PROJECT_ROOT / "data" / "processed"

for morphology_folder in roboflow_folder.iterdir():
    if not morphology_folder.is_dir(): #ignore anything that isn't a folder
        continue 

    input_folder = morphology_folder / 'train'
    output_folder = mask_folder / morphology_folder.name 

    # from coco -> mask
    with open(input_folder / '_annotations.coco.json', 'r') as file:
        coco = json.load(file)
    images = {}
    for image in coco['image']:
        images[image['id']] = image
    for image_id, image_info in images.items():
        height = image_info['height']
        width = image_info['width']
        # blank mask
        mask = np.zeros((height,width), dtype=np.uint8)

        for ann in coco['annotations']:
            if ann['image_id'] != image_id:
                continue

            for polygon in ann['segmentation']:
                points = np.array(polygon).reshape(-1,2)
                points = points.astype(np.int32)

                cv2.fillPoly(mask, [points], 255)

        filename = Path(image_info['file_name']).stem + '.png'
        cv2.imwrite(str(output_folder / filename), mask)

print('fin')