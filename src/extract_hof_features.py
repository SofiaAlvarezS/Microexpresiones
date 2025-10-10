# src/extract_hof_features.py
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml
from src.optical_flow import calc_flows_gray_sequence
from src.window_buffer import SlidingWindowBuffer

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_hof_from_sequence(frames, num_bins=16):
    """Calcula un descriptor HOF promedio para una secuencia de frames (ventana temporal)."""
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    flows = calc_flows_gray_sequence(frames_gray)

    mags, angs = [], []
    for flow in flows:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(mag)
        angs.append(ang)

    if not mags:
        return np.zeros(num_bins, dtype=np.float32)

    mags = np.stack(mags)
    angs = np.stack(angs)
    hist = cv2.calcHist([angs.astype(np.float32)], [0], None, [num_bins], [0, 2 * np.pi])
    hist = hist.flatten()
    hist /= (np.sum(hist) + 1e-7)
    return hist


def extract_features_from_roi_folder(roi_dir, num_bins=16, window_size=16, stride=8):
    """Extrae HOF de una ROI completa (carpeta con imágenes)."""
    image_files = sorted(list(roi_dir.glob("*.png")) + list(roi_dir.glob("*.jpg")))
    if len(image_files) < window_size:
        return [], []

    buffer = SlidingWindowBuffer(window_size=window_size, stride=stride)
    features, labels = [], []

    for img_path in tqdm(image_files, desc=f"🧩 {roi_dir.name}", leave=False, ncols=90):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 🔍 Extraer etiqueta del nombre del archivo (ej: "Anger_0000_cejas.png")
        label = img_path.stem.split("_")[0]

        windows = buffer.add_frame(img)
        for win in windows:
            hof_vec = extract_hof_from_sequence(win, num_bins=num_bins)
            features.append(hof_vec)
            labels.append(label)

    return features, labels


def extract_all_hof_features(cfg):
    rois_root = Path(cfg["dataset"]["rois_path"])
    features_output = Path(cfg["features"]["output_file"])
    num_bins = cfg["features"]["bins"]

    all_features, all_labels = [], []

    print("\n🚀 Iniciando extracción de HOF...\n")

    for dataset_dir in rois_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        print(f"📂 Dataset: {dataset_dir.name}")

        for roi_dir in sorted(dataset_dir.iterdir()):
            if not roi_dir.is_dir():
                continue

            print(f"🧠 Procesando ROI: {roi_dir.name}")

            features, labels = extract_features_from_roi_folder(
                roi_dir, num_bins=num_bins
            )
            if features:
                all_features.extend(features)
                all_labels.extend(labels)

    print("\n💾 Guardando features...")
    df = pd.DataFrame(all_features)
    df["label"] = all_labels
    features_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_output, index=False)
    print(f"✅ Extracción completa. Archivo: {features_output}")

    print("\n📊 Resumen de etiquetas:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    cfg = load_config()
    extract_all_hof_features(cfg)
