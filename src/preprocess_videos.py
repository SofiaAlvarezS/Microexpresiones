import cv2
import yaml
from pathlib import Path
from utils import create_face_mesh, extract_regions

def process_dataset(ds_root, dataset_name, cfg, face_mesh):
    """
    Procesa un dataset (SAMM o CASME II) extrayendo ROIs.
    """
    rois_root = Path(cfg["dataset"]["rois_path"])
    roi_size = tuple(cfg["dataset"]["roi_size"])
    exts = [e.lower() for e in cfg["io"]["image_extensions"]]

    label_map = cfg.get("label_mapping", {}).get(dataset_name, {})

    for class_dir in sorted(ds_root.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        mapped_label = label_map.get(label, label)

        out_dir = rois_root / dataset_name / mapped_label
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in exts:
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            regs = extract_regions(frame, face_mesh)
            if not regs:
                print(f"⚠️ No se detectó rostro en {img_path}")
                continue

            for roi_name, roi in regs.items():
                roi = cv2.resize(roi, roi_size)
                out_file = out_dir / f"{img_path.stem}_{roi_name}.png"
                cv2.imwrite(str(out_file), roi)


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset_roots = cfg["dataset"]["raw_paths"]
    face_mesh = create_face_mesh()

    for ds_root in dataset_roots:
        ds_root = Path(ds_root)
        if not ds_root.exists():
            print(f"⚠️ {ds_root} no existe, se omite")
            continue

        dataset_name = ds_root.name
        print(f"📂 Procesando dataset: {dataset_name}")
        process_dataset(ds_root, dataset_name, cfg, face_mesh)

    face_mesh.close()
    print("✅ Preprocesamiento completado.")


if __name__ == "__main__":
    main()
