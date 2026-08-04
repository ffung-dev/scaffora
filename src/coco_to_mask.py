# import
import json
from pathlib import Path
import cv2
import numpy as np

roboflow_folder = Path('../data/masks_roboflow')
mask_folder = Path('../data/processed')

for morphology_folder in roboflow_folder.iterdir():
    if not morphology_folder.is_dir(): #ignore anything that isn't a folder
        continue 
    