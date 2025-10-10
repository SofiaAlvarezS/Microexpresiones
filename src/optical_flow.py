import cv2
import numpy as np
import yaml

# Cachear el modo para no imprimir en cada frame
_MODE_CACHE = None

def load_mode_from_config():
    """Lee el modo (fast/full) desde config.yaml, con cache."""
    global _MODE_CACHE
    if _MODE_CACHE is not None:
        return _MODE_CACHE
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _MODE_CACHE = cfg.get("features", {}).get("mode", "fast")
    except Exception:
        _MODE_CACHE = "fast"
    return _MODE_CACHE


def calc_flows_gray_sequence(frames_gray):
    """
    Calcula flujos ópticos entre frames consecutivos.
    - "fast": usa Farnebäck (más rápido)
    - "full": usa TV-L1 (más preciso)
    Corrige tamaños inconsistentes y descarta frames corruptos.
    """
    flows = []
    if len(frames_gray) < 2:
        return flows

    mode = load_mode_from_config()

    # Filtrar frames inválidos
    frames_gray = [f for f in frames_gray if f is not None and f.size > 0]
    if len(frames_gray) < 2:
        return flows

    base_h, base_w = frames_gray[0].shape[:2]
    frames_gray = [cv2.resize(f, (base_w, base_h)) for f in frames_gray]

    if mode == "full":
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        for i in range(len(frames_gray) - 1):
            prev, nxt = frames_gray[i], frames_gray[i + 1]
            flow = TVL1.calc(prev, nxt, None)
            flows.append(flow)
    else:
        for i in range(len(frames_gray) - 1):
            prev, nxt = frames_gray[i], frames_gray[i + 1]
            if prev.shape != nxt.shape:
                nxt = cv2.resize(nxt, (prev.shape[1], prev.shape[0]))

            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flows.append(flow)

    return flows
