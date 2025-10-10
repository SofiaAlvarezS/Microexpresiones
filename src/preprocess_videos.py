import cv2
import yaml
from pathlib import Path
from utils import create_face_mesh, extract_regions


def load_config():
    """Carga el archivo de configuración config.yaml"""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_image_folder(folder_path, output_root, face_mesh):
    """
    Procesa una carpeta con imágenes (como una secuencia de frames).
    Detecta el rostro, extrae ROIs y las guarda organizadas.
    """
    # Tomar todas las imágenes válidas (.jpg o .png)
    images = sorted([p for p in Path(folder_path).glob("*") if p.suffix.lower() in [".jpg", ".png"]])
    if not images:
        print(f"⚠️ No se encontraron imágenes en {folder_path}")
        return

    folder_name = Path(folder_path).name
    dataset_name = Path(folder_path).parent.name
    total_saved = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠️ No se pudo leer {img_path}")
            continue

        # Detectar y extraer regiones faciales
        regions = extract_regions(frame, face_mesh)
        if not regions:
            print(f"⚠️ No se detectó rostro en {folder_name}/{img_path.name}")
            continue

        # Guardar cada región en su carpeta correspondiente
        for roi_name, roi_img in regions.items():
            out_dir = output_root / dataset_name / roi_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{folder_name}_{idx:04d}_{roi_name}.png"
            cv2.imwrite(str(out_dir / out_name), roi_img)
            total_saved += 1

    print(f"✅ {folder_name}: {len(images)} imágenes procesadas, {total_saved} ROIs guardadas.")


def preprocess_all_datasets(config):
    """
    Recorre todas las carpetas definidas en config.yaml
    y procesa las secuencias de imágenes dentro de cada dataset.
    """
    dataset_roots = config["dataset"]["videos_path"]
    rois_root = Path(config["dataset"]["rois_path"])
    face_mesh = create_face_mesh()

    for dataset_path in dataset_roots:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            print(f"⚠️ No se encontró la carpeta {dataset_dir}")
            continue

        print(f"\n📂 Procesando dataset: {dataset_dir.name}")

        # Cada subcarpeta dentro del dataset corresponde a una secuencia
        for subfolder in sorted(dataset_dir.glob("*")):
            if subfolder.is_dir():
                preprocess_image_folder(subfolder, rois_root, face_mesh)

    face_mesh.close()
    print("\n✅ Preprocesamiento completado.")


if __name__ == "__main__":
    cfg = load_config()
    preprocess_all_datasets(cfg)
