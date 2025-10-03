# src/utils.py
import cv2
import numpy as np
import mediapipe as mp


# =========================
# FACE MESH HANDLER
# =========================
def create_face_mesh():
    """Inicializa el detector FaceMesh de MediaPipe."""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


# =========================
# REGIONES DE INTERÉS
# =========================
ROI_LANDMARKS = {
    "ojos_izq": list(range(33, 133)),   # aproximado ojo izquierdo
    "ojos_der": list(range(362, 463)),  # aproximado ojo derecho
    "cejas": list(range(65, 295)),      # cejas y frente
    "nariz_boca": list(range(1, 18)) + list(range(61, 291)),  # nariz + boca
    "rostro_completo": list(range(0, 468)),  # toda la cara
}


def extract_regions(frame, face_mesh):
    """Extrae las regiones faciales definidas en ROI_LANDMARKS."""
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    regs = {}

    for name, idxs in ROI_LANDMARKS.items():
        coords = [(int(face_landmarks.landmark[i].x * w),
                   int(face_landmarks.landmark[i].y * h)) for i in idxs]
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)

        # recorte con margen
        x_min, y_min = max(0, x_min - 5), max(0, y_min - 5)
        x_max, y_max = min(w, x_max + 5), min(h, y_max + 5)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            regs[name] = roi

    return regs


# =========================
# FEATURES: HISTOGRAMA HOG-OPTICAL FLOW SIMPLIFICADO
# =========================
def compute_optical_flow_histogram(gray_img, bins=16):
    """
    Calcula histograma de orientaciones basado en gradientes (HOG simple).
    Devuelve un vector normalizado de longitud = bins.
    """
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    hist, _ = np.histogram(ang, bins=bins, range=(0, 360), weights=mag)
    hist = hist.astype("float32")
    hist /= (np.sum(hist) + 1e-6)  # normalización
    return hist


# =========================
# HELPERS
# =========================
def prepare_roi(roi, size=(128, 128)):
    """Preprocesa ROI: resize, escala a gris y normaliza."""
    roi_resized = cv2.resize(roi, size)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    return gray
