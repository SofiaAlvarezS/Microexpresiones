# src/inspect_samm.py
import cv2
from pathlib import Path
from utils import create_face_mesh

# Ajusta esta ruta según necesites
root = Path("data_raw/SAMM_v1/Disgust")

# ROIs: listas de índices de MediaPipe FaceMesh (468 puntos).
ROI_LANDMARKS = {
    "ojo_izq": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "ojo_der": [362, 382, 381, 380, 374, 373, 390, 249, 263],
    "boca":    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
    "nariz":   [1, 2, 98, 327, 168],
    "rostro_completo": list(range(0, 468)),
}

# colores para diferenciar (B,G,R)
COLORS = {
    "ojo_izq": (255, 0, 0),
    "ojo_der": (0, 0, 255),
    "boca": (0, 255, 0),
    "nariz": (0, 255, 255),
    "rostro_completo": (255, 255, 255),
}

# padding en pixeles relativo al tamaño de la cara (ajustable)
PAD_PCT = 0.08  # 8% del max(width,height)

def landmarks_to_coords(landmarks, w, h):
    """Convierte la lista de landmarks de MediaPipe a coordenadas (x,y)."""
    coords = []
    for lm in landmarks.landmark:
        coords.append((int(lm.x * w), int(lm.y * h)))
    return coords

def bbox_from_indices(coords, indices, pad_px, img_w, img_h):
    pts = [coords[i] for i in indices if 0 <= i < len(coords)]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = max(0, min(xs) - pad_px)
    y_min = max(0, min(ys) - pad_px)
    x_max = min(img_w, max(xs) + pad_px)
    y_max = min(img_h, max(ys) + pad_px)
    return (x_min, y_min, x_max, y_max)

def main():
    face_mesh = create_face_mesh()

    for img_path in sorted(root.glob("*.jpg"))[:30]:  # prueba con hasta 30 imágenes
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("No se pudo leer:", img_path)
            continue

        h, w = frame.shape[:2]
        pad_px = int(max(w, h) * PAD_PCT)

        # procesar con MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print(f"❌ No detectó rostro en {img_path.name}")
            cv2.imshow("Regiones (ninguna detectada)", frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
            continue

        coords = landmarks_to_coords(results.multi_face_landmarks[0], w, h)

        # dibujar rectángulos calculados por indices
        overlay = frame.copy()
        for name, indices in ROI_LANDMARKS.items():
            bbox = bbox_from_indices(coords, indices, pad_px, w, h)
            if bbox:
                x0, y0, x1, y1 = bbox
                color = COLORS.get(name, (255, 255, 255))
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

        # mostrar sólo la imagen con divisiones
        cv2.imshow("Regiones faciales (solo cajas)", overlay)
        key = cv2.waitKey(0)
        if key & 0xFF == ord("q"):
            break

    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
