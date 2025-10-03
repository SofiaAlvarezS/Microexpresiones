# src/inspect_casme.py
import cv2
from pathlib import Path
from utils import create_face_mesh
import numpy as np

root = Path("data_raw/CASME2/Happiness")  # ajusta la carpeta de emoción que quieras testear
face_mesh = create_face_mesh()

# landmarks frontales para "frente"
ROI_LANDMARKS_FRONT = [10, 67, 103, 109, 338, 297, 332, 284]

for img_path in sorted(root.glob("*.jpg"))[:10]:  # prueba con 10 imágenes
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"⚠️ No pude leer {img_path}")
        continue

    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print(f"❌ No detectó rostro en {img_path.name}")
        continue

    face_landmarks = results.multi_face_landmarks[0]

    # regiones de interés
    regiones = {
        "rostro_completo": list(range(0, 468)),
        "ojo_izq": list(range(362, 463)),
        "ojo_der": list(range(33, 133)),
        "boca": list(range(78, 308)),
        "frente": ROI_LANDMARKS_FRONT
    }

    colores = {
        "rostro_completo": (255, 255, 255),  # blanco
        "ojo_izq": (255, 0, 0),              # azul
        "ojo_der": (0, 0, 255),              # rojo
        "boca": (0, 255, 0),                 # verde
        "frente": (0, 255, 255)              # amarillo
    }

    for nombre, idxs in regiones.items():
        coords = [(int(face_landmarks.landmark[i].x * w),
                   int(face_landmarks.landmark[i].y * h)) for i in idxs]
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        x_min, y_min = max(0, x_min - 5), max(0, y_min - 5)
        x_max, y_max = min(w, x_max + 5), min(h, y_max + 5)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colores[nombre], 2)

    cv2.imshow("Regiones CASME-II", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
face_mesh.close()
