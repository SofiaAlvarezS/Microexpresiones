from pathlib import Path
import cv2
import numpy as np
from src.optical_flow import calc_flows_gray_sequence
from src.window_buffer import SlidingWindowBuffer

def main():
    # Ruta de prueba (ajústala según tu dataset)
    test_folder = Path("data_raw/CASME2/happiness")
    image_paths = sorted([p for p in test_folder.glob("*.jpg")])

    if not image_paths:
        print(f"❌ No se encontraron imágenes en {test_folder}")
        return

    print(f"Encontradas {len(image_paths)} imágenes para prueba.")

    # Inicializar buffer (por si lo necesitas luego para HOF)
    buffer = SlidingWindowBuffer(window_size=5, stride=1)

    # Leer imágenes en escala de grises
    frames_gray = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames_gray.append(img)

    # Calcular flujo óptico
    flows = calc_flows_gray_sequence(frames_gray, mode="FAST")
    print(f"✅ Flujo óptico calculado para {len(flows)} pares de frames.")

    # Visualizar el primer flujo
    if flows:
        flow = flows[0]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2          # Hue = dirección
        hsv[..., 1] = 255                            # Saturación máxima
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Valor = magnitud

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Flujo óptico (primer par de frames)", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("⚠️ No se pudo calcular el flujo óptico.")

if __name__ == "__main__":
    main()
