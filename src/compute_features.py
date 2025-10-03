import cv2
import yaml
import pandas as pd
from pathlib import Path
from utils import compute_optical_flow_histogram

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    rois_root = Path(cfg["dataset"]["rois_path"])
    bins = cfg["features"]["bins"]
    output_file = Path(cfg["features"]["output_file"])
    save_per_dataset = cfg["features"].get("save_per_dataset", False)

    all_features = []

    for dataset_dir in rois_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        print(f"🔎 Extrayendo features de {dataset_name}")

        ds_features = []

        for label_dir in dataset_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name

            for roi_path in label_dir.glob("*.png"):
                gray = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue

                feat = compute_optical_flow_histogram(gray, bins=bins)
                row = {
                    "dataset": dataset_name,
                    "file": str(roi_path),
                    "label": label,
                    **{f"f{i}": v for i, v in enumerate(feat)}
                }
                ds_features.append(row)
                all_features.append(row)

        if save_per_dataset and ds_features:
            df_ds = pd.DataFrame(ds_features)
            out_ds_file = output_file.parent / f"{dataset_name}_features.csv"
            df_ds.to_csv(out_ds_file, index=False)
            print(f"✅ Features guardadas en {out_ds_file} ({len(df_ds)} muestras)")

    if not all_features:
        print("⚠️ No se generaron features")
    else:
        df = pd.DataFrame(all_features)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✅ Features combinadas guardadas en {output_file} ({len(df)} muestras)")


if __name__ == "__main__":
    main()
