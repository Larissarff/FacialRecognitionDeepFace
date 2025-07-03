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

dfs = DeepFace.find(
    img_path="DataFrame/tests/teste_03.jpg",
    db_path="DataFrame/data",
    enforce_detection=False,
    model_name=models[0],
    detector_backend=backends[7],
)

for x in range(len(dfs)):
    print(dfs[x])

