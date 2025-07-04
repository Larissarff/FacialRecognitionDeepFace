from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

image = "DataFrame/data/Beyonce"

DeepFace.stream(db_path=image)