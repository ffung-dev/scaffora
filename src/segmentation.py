# import
import cv2
import numpy as np
from pathlib import Path
raw_folder = Path('../data/raw')
processed_folder = Path('../data/processed')

def edit_image(image_path):

    image = cv2.imread(str(image_path))
    # resize
    height, width = image.shape[:2]
    scale = 1024 / width

    image = cv2.resize(image, (1024, int(height * scale)))
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur to smooth edges
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    return blur

def threshold(edited_image):

    _, thresh = cv2.threshold(edited_image, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    # cleaning
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def largest_comp(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    largest_c = max(contours, key=cv2.contourArea)
    # draw largest
    stencil = np.zeros_like(mask)
    cv2.drawContours(stencil, [largest_c], 0, 255, -1)
    largest_mask = cv2.bitwise_and(mask, stencil)
    return largest_mask

for morphology in raw_folder.iterdir(): # each type of branching
    if not morphology.is_dir(): # not folder
        continue
    output_folder = processed_folder / morphology.name
    # segment each image
    for image_path in morphology.iterdir():
        edited_image = edit_image(image_path)
        mask = threshold(edited_image)
        mask = largest_comp(mask)
        cv2.imwrite(output_folder, mask)
    



