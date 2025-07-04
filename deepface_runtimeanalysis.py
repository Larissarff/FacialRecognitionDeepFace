from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

DeepFace.stream(db_path="DataFrame/data/larissa")