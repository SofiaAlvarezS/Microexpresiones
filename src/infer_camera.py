# src/infer_camera.py
import cv2
import joblib
import numpy as np
from utils import create_face_mesh, extract_regions, prepare_roi, compute_optical_flow_histogram

def main():
    # Cargar modelo entrenado
    clf = joblib.load("models/svm_model.joblib")
    print("✅ Modelo cargado desde models/svm_model.joblib")

    # Inicializar FaceMesh
    face_mesh = create_face_mesh()

    # Abrir cámara
    cap = cv2.VideoCapture(0)

    bins = 16  # debe coincidir con compute_features.py
    roi_size = (128, 128)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        regs = extract_regions(frame, face_mesh)
        if regs is None:
            cv2.imshow("Microexpresiones", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Usamos solo el ROI completo
        roi = regs["rostro_completo"]
        gray_roi = prepare_roi(roi, size=roi_size)

        # Extraer features del frame actual
        feat = compute_optical_flow_histogram(gray_roi, bins=bins)
        X = feat.reshape(1, -1)

        # Inferencia
        pred = clf.predict(X)[0]

        # Mostrar en pantalla
        cv2.putText(frame, f"Pred: {pred}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Microexpresiones", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
