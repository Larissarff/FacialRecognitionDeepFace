from deepface import DeepFace
import cv2
import tempfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

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

def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw the rectangle with rounded corners
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)



image = "DataFrame/data/larissa/larissa_03.jpeg"
path = "DataFrame/data/larissa"

# Abrir a câmera com resolução 640x480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Definir tamanhos desejados para exibição
display_width = 320
display_height = 240

cv2.namedWindow("Câmera", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o frame da câmera.")
        break

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, frame)
        temp_file_path = temp_file.name

        # Usar o conteúdo do arquivo temporário para análise
        results = DeepFace.find(img_path=temp_file_path, db_path=path, enforce_detection=False)

    os.remove(temp_file_path)

DeepFace.stream(db_path=path, camera=cap)