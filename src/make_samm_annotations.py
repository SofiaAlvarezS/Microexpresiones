# src/make_samm_annotations.py
import os
import csv
from pathlib import Path
from collections import defaultdict

DEFAULT_ROIS = ["rostro_completo", "ojo_izq", "ojo_der", "frente", "nariz_boca"]

def make_annotations(dataset_root: str = "data_raw/SAMM_v1",
                     output_csv: str = "annotations/samm_annotations.csv",
                     roi_list=None):
    roi_list = roi_list or DEFAULT_ROIS
    rows = []
    dataset_root = Path(dataset_root)

    # Agrupamos imágenes por prefijo antes del "_" → ej: "011"
    for emotion_dir in sorted(dataset_root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        files = sorted(emotion_dir.glob("*.jpg"))
        groups = defaultdict(list)
        for f in files:
            prefix = f.stem.split("_")[0]  # ej: "011" de "011_1220"
            groups[prefix].append(f)

        for video_id, frames in groups.items():
            start = 0
            end = len(frames) - 1
            for roi in roi_list:
                rows.append([video_id, start, end, emotion_dir.name, roi, 'unknown'])

    os.makedirs(Path(output_csv).parent, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'start_frame', 'end_frame', 'label', 'roi_name', 'subject_id'])
        writer.writerows(rows)

    print(f"✅ CSV generado en {output_csv} con {len(rows)} filas")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data_raw/SAMM_v1')
    ap.add_argument('--out', default='annotations/samm_annotations.csv')
    args = ap.parse_args()
    make_annotations(args.dataset, args.out)
