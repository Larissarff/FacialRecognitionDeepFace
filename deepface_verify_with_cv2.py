from deepface import DeepFace
import cv2
import tempfile
import os
import time
import numpy as np
import mediapipe as mp

# ---------- Configurações ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
image = "DataFrame/data/larissa/larissa_02.jpeg"  # imagem base
check_interval = 15
min_ear_threshold = 0.2
consecutive_frames_to_blink = 2

# ---------- Mediapipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ---------- EAR utility ----------
def calculate_EAR(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Olhos: pontos com base nos índices do Mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye_landmarks(face_landmarks, image_shape):
    ih, iw = image_shape
    left_eye = np.array([(face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih) for i in LEFT_EYE])
    right_eye = np.array([(face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih) for i in RIGHT_EYE])
    return left_eye, right_eye

# ---------- Captura ----------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cv2.namedWindow("Câmera", cv2.WINDOW_NORMAL)

last_checked = time.time()
blink_counter = 0
blink_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura de frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            left_eye, right_eye = get_eye_landmarks(face_landmarks, (ih, iw))

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < min_ear_threshold:
                blink_counter += 1
            else:
                if blink_counter >= consecutive_frames_to_blink:
                    blink_detected = True
                    print("[👁️ Piscada detectada! Verificação liberada.]")
                blink_counter = 0

            # Visualização (opcional)
            for point in left_eye:
                cv2.circle(frame, tuple(np.int32(point)), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, tuple(np.int32(point)), 2, (0, 255, 0), -1)

    # Só verifica rosto após piscada e intervalo
    current_time = time.time()
    if blink_detected and current_time - last_checked >= check_interval:
        last_checked = current_time
        blink_detected = False  # Reinicia verificação

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, frame)
            temp_file_path = temp_file.name

        try:
            result = DeepFace.verify(
                img1_path=image,
                img2_path=temp_file_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,
                anti_spoofing=True,
            )

            verified = result["verified"]
            distance = result["distance"]

            if verified:
                print(f"[✅ MESMA PESSOA] Distância: {distance:.4f}")
            else:
                print(f"[❌ PESSOA DIFERENTE] Distância: {distance:.4f}")

        except Exception as e:
            print("Erro ao verificar:", e)

        os.remove(temp_file_path)

    cv2.imshow("Câmera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()