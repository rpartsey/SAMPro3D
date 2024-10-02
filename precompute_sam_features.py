import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

from segment_anything import sam_model_registry, SamPredictor


def precompute_sam_features(predictor, image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    features = predictor.features.cpu().numpy()
    np.save(output_path, features)


device = torch.device("cuda:0")
sam = sam_model_registry["vit_h"]("sam_vit_h_4b8939.pth").to(device=device)
predictor = SamPredictor(sam)


scenes = [
    "scene0000_02", 
    "scene0008_00", 
    "scene0013_01", 
    "scene0019_00", 
    "scene0025_00", 
    "scene0031_00", 
    "scene0044_00", 
    "scene0062_00", 
    "scene0067_00"
]

for scene in scenes:
    print(scene)

    frames_path = os.path.join("/home/rpartsey/code/eai/SAMPro3D-fork/SAMPro3D/data/scannet/scans", scene, "color")
    os.makedirs(frames_path.replace("color", "features"), exist_ok=True)

    for frame in tqdm(sorted(os.listdir(frames_path))):
        image_path = os.path.join(frames_path, frame)
        output_path = image_path.replace("color", "features").replace(".jpg", ".npy")

        precompute_sam_features(predictor, image_path, output_path)
