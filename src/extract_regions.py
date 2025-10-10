import cv2
import numpy as np

def extract_regions(frame, face_mesh, roi_size=(96, 96)):
    """
    Extrae regiones de interés (ROIs) del rostro: ojos, cejas, nariz+boca, rostro completo.
    Retorna un diccionario {nombre_roi: imagen_roi}.
    """
    regions = {}
    h, w, _ = frame.shape

    results = face_mesh.process(frame)
    if not results.multi_face_landmarks:
        return regions

    landmarks = results.multi_face_landmarks[0].landmark

    def get_bbox(indices, padding=0.1):
        pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
        x, y, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        dx, dy = int((x2 - x) * padding), int((y2 - y) * padding)
        x, y = max(x - dx, 0), max(y - dy, 0)
        x2, y2 = min(x2 + dx, w), min(y2 + dy, h)
        return (x, y, x2, y2)

    def crop_region(bbox):
        x, y, x2, y2 = bbox
        region = frame[y:y2, x:x2]
        if region.size == 0:
            return None
        return cv2.resize(region, roi_size)

    # --- Regiones principales basadas en landmarks de MediaPipe ---
    # Ojo izquierdo
    left_eye_idx = list(range(33, 133))
    # Ojo derecho
    right_eye_idx = list(range(362, 463))
    # Cejas (aprox)
    brow_idx = list(range(70, 103))
    # Nariz y boca
    mouth_nose_idx = list(range(1, 17)) + list(range(164, 200))
    # Rostro completo
    full_face_idx = list(range(0, 468))

    regions["ojos_izq"] = crop_region(get_bbox(left_eye_idx))
    regions["ojos_der"] = crop_region(get_bbox(right_eye_idx))
    regions["cejas"] = crop_region(get_bbox(brow_idx))
    regions["nariz_boca"] = crop_region(get_bbox(mouth_nose_idx))
    regions["rostro_completo"] = crop_region(get_bbox(full_face_idx))

    return regions
