from deepface import DeepFace

models = [
    "VGG-Face",
    "Facenet",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]
backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "centerface",
]

result = DeepFace.verify(
    img1_path = 'DataFrame/data/Eminem/eminem_04.jpg',
    img2_path = 'DataFrame/data/Eminem/eminem_05.jpg',
    model_name = models[0],
    enforce_detection = False, # nao quero que force detecção em rostos
    detector_backend = backends[7],
)

print(result)