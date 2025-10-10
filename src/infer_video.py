import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import Counter
from src.extract_regions import extract_regions
from src.optical_flow import calc_flows_gray_sequence
from src.extract_hof_features import extract_hof_from_sequence
from src.window_buffer import SlidingWindowBuffer


def preprocess_roi(roi, target_size=(128, 128)):
    """Aplica CLAHE, convierte a gris y redimensiona al tamaño usado en entrenamiento."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    resized = cv2.resize(gray_eq, target_size)
    return resized


def run_inference_video(video_path):
    print(f"🎬 Analizando video: {video_path}")
    print("📦 Cargando modelo desde: models/svm_model.joblib")

    # Cargar modelo
    model_bundle = joblib.load("models/svm_model.joblib")
    if hasattr(model_bundle, "predict"):
        clf = model_bundle
        scaler = None
    else:
        clf = model_bundle["model"]
        scaler = model_bundle.get("scaler", None)

    # Inicializar MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 FPS: {fps:.2f} | Total de frames: {total_frames}")

    buffer = SlidingWindowBuffer(window_size=10, stride=2)
    preds = []
    processed = 0

    print("\n🚀 Procesando video (presiona Q para salir)...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1

        # Mostrar el video sin texto de emoción
        cv2.imshow("Inferencia (sin resultados en vivo)", frame)

        # Procesamiento del frame (sin mostrar emoción)
        regions = extract_regions(frame, face_mesh)
        rostro = regions.get("rostro_completo")
        if rostro is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        roi_pre = preprocess_roi(rostro)
        windows = buffer.add_frame(roi_pre)
        if not windows:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        hof_vec = extract_hof_from_sequence(windows[0])
        hof_vec = np.array(hof_vec).reshape(1, -1)
        if scaler is not None:
            hof_vec = scaler.transform(hof_vec)

        pred = clf.predict(hof_vec)[0]
        preds.append(pred)

        if processed % 100 == 0:
            progress = (processed / total_frames) * 100
            print(f"🧩 Progreso: {progress:.1f}% ({processed}/{total_frames})")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ───────────────────────────────
    # 📊 Mostrar estadísticas finales
    # ───────────────────────────────
    if not preds:
        print("⚠️ No se generaron predicciones.")
        return

    print("\n✅ Procesamiento completado.")
    print(f"🖼️ Frames analizados: {len(preds)}")

    counter = Counter(preds)
    total_preds = sum(counter.values())
    dominant, count = counter.most_common(1)[0]

    print("\n📈 Estadísticas finales:")
    print("───────────────────────────")
    for emotion, c in counter.most_common():
        percent = (c / total_preds) * 100
        print(f"{emotion:<15} → {c:>5} frames ({percent:>5.1f}%)")

    print("───────────────────────────")
    print(f"🎭 Emoción dominante: {dominant} ({(count / total_preds) * 100:.1f}%)")

    # 💾 Guardar resultados
    import csv
    output_csv = "models/video_inference_summary.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "predicted_emotion"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

    print(f"\n💾 Resultados guardados en: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inferencia de microexpresiones en video (resumen final).")
    parser.add_argument("--path", type=str, required=True, help="Ruta del video a analizar")
    args = parser.parse_args()

    run_inference_video(args.path)
