import os
import cv2
import json
from PIL import Image
from deepface import DeepFace

# Detect face
img1_path = "data/derek1.jpg"
img2_path = "data/derek3.jpg"

result = DeepFace.verify(
  img1_path = img1_path,
  img2_path = img2_path,
)
print(json.dumps(result, indent = 2))

# Find face
dfs = DeepFace.find(
  img_path = "data/derek1.jpg",
  db_path = "data",
)
print(dfs)
