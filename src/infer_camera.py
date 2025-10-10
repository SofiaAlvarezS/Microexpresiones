import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time
from src.extract_regions import extract_regions
from src.extract_hof_features import extract_hof_from_sequence
from src.optical_flow import calc_flows_gray_sequence
from src.window_buffer import SlidingWindowBuffer

# --- CONFIGURACIÓN ---
ROI_SIZE = (128, 128)
WINDOW_SIZE = 5           # cantidad de frames por ventana
SMOOTHING_WINDOW = 15     # predicciones para suavizado

# --- CARGAR MODELO ---
print("✅ Cargando modelo...")
pipeline = joblib.load("models/svm_model.joblib")
print("✅ Modelo cargado correctamente (Pipeline detectado).")

# --- Inicializar Mediapipe FaceMesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# --- Buffers ---
frame_buffer = SlidingWindowBuffer(window_size=WINDOW_SIZE, stride=1)
pred_queue = deque(maxlen=SMOOTHING_WINDOW)
emotion_counter = Counter()

# --- Captura de cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo acceder a la cámara.")

print("🎥 Cámara iniciada. Presiona 'q' para salir.")
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # --- Extraer regiones faciales ---
    regions = extract_regions(frame, face_mesh)
    if "rostro_completo" not in regions:
        cv2.putText(frame, "No se detecta rostro", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Microexpresiones - Tiempo Real", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    roi = cv2.resize(regions["rostro_completo"], ROI_SIZE)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    windows = frame_buffer.add_frame(gray)

    if windows:
        seq = windows[-1]
        hof_vec = extract_hof_from_sequence(seq)
        if hof_vec is not None:
            pred = pipeline.predict([hof_vec])[0]
            pred_queue.append(pred)
            emotion_counter[pred] += 1

    # --- Suavizado y visualización ---
    if len(pred_queue) > 0:
        pred_counts = Counter(pred_queue)
        stable_pred = pred_counts.most_common(1)[0][0]
        cv2.putText(frame, f"Emocion: {stable_pred}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Microexpresiones - Tiempo Real", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
end_time = time.time()

# --- Mostrar estadísticas ---
print("\n📊 Estadísticas de sesión:")
print(f"⏱️  Duración: {end_time - start_time:.2f} segundos")
print(f"🎞️  Frames procesados: {frame_count}")
print(f"🧠  Emociones detectadas: {sum(emotion_counter.values())}")
if emotion_counter:
    print("📈 Frecuencia de emociones:")
    for emo, count in emotion_counter.most_common():
        print(f"   - {emo}: {count} veces")
    dominant = emotion_counter.most_common(1)[0][0]
    print(f"\n💫 Emoción predominante: {dominant}")
else:
    print("⚠️ No se detectaron emociones durante la sesión.")
