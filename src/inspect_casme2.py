# src/inspect_casme2.py
import cv2
from pathlib import Path
from utils import create_face_mesh, extract_regions

# Ajusta la ruta según tu estructura de CASME2
# Por ejemplo, si tienes: data_raw/CASME_II/Disgust/
root = Path("data_raw/CASME_II/Disgust")

# Inicializar detector FaceMesh
face_mesh = create_face_mesh()

# Tomar hasta 10 imágenes de prueba
for img_path in sorted(root.glob("*.jpg"))[:10]:
    frame = cv2.imread(str(img_path))
    regs = extract_regions(frame, face_mesh)
    if regs is None:
        print(f"❌ No detectó rostro en {img_path.name}")
        continue

    for name, roi in regs.items():
        cv2.imshow(name, roi)
    cv2.waitKey(0)

cv2.destroyAllWindows()
face_mesh.close()
