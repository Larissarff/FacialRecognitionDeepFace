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
    img1_path = 'DataFrame/data/rihana_01.jpg',
    img2_path = 'DataFrame/data/rihana_02.jpg',
    model_name = models[0],
    enforce_detection = False, # nao quero que force detecção em rostos
    detector_backend = backends[7],
)
# quanto menor for a "distance" entre uma imagem e outra, mais provavel que seja a mesma pessoa
# se 'threshold' for 0.4, significa que se a distância for menor que 0.4, é a mesma pessoa

print(result)