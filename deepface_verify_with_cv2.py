from deepface import DeepFace
import cv2
import tempfile
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forçar uso da CPU

image = "DataFrame/data/larissa/larissa_03.jpeg"  # imagem de referência

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Câmera", cv2.WINDOW_NORMAL)

last_checked = time.time()
check_interval = 15 # intervalo de verificação em segundos

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o frame da câmera.")
        break

    current_time = time.time()
    if current_time - last_checked >= check_interval:
        last_checked = current_time

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, frame)
            temp_file_path = temp_file.name

        try:
            results = DeepFace.verify(
                img1_path=image,
                img2_path=temp_file_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,
                anti_spoofing=True,
            )

            distance = results["distance"]
            verified = results["verified"]

            if verified:
                print(f"[✅ MESMA PESSOA] Distância: {distance:.4f}")
            else:
                print(f"[❌ PESSOA DIFERENTE] Distância: {distance:.4f}")

        except Exception as e:
            print("Erro durante a verificação:", e)

        os.remove(temp_file_path)

    # Mostrar a imagem da câmera
    cv2.imshow("Câmera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()