# import and install
import cv2
import matplotlib.pyplot as plt
import torch

# setup sam
CHECKPOINT_PATH = "../models/sam_vit_b_01ec64.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
print(MODEL_TYPE)
print(CHECKPOINT_PATH)

# resize image
image = cv2.imread("../data/raw/arborescent/abrolhosensis1.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = rgb.shape[:2]
scale = 1024/width
rgb = cv2.resize(rgb, (1024, (int)(height * scale)))

# get points
fig, ax = plt.subplots()
ax.imshow(rgb)
points = plt.ginput(n=5, timeout=0, show_clicks=True)