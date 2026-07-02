import cv2
import numpy as np

def aplicar_segmentacion(frame_bgr, bbox):
    """
    Función de respaldo para la segmentación y detección de anomalías (botella sucia).
    """
    frame_copia = frame_bgr.copy()
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_copia.shape[1], x2), min(frame_copia.shape[0], y2)

    if (x2 <= x1) or (y2 <= y1):
        return frame_copia, "OPTIMO"

    roi = frame_copia[y1:y2, x1:x2]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_dark = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([179, 255, 80]))
    mask_sat = cv2.inRange(roi_hsv, np.array([0, 100, 0]), np.array([179, 255, 255]))
    mask_anomalia = cv2.bitwise_or(mask_dark, mask_sat)

    total_pixels = roi.shape[0] * roi.shape[1]
    anomalia_pixels = np.count_nonzero(mask_anomalia)
    porcentaje = anomalia_pixels / total_pixels if total_pixels > 0 else 0.0

    estado_limpieza = "OPTIMO" if porcentaje < 0.10 else "ANOMALIA"

    anomalia_bgr = np.zeros_like(roi)
    anomalia_bgr[mask_anomalia > 0] = [0, 0, 255]

    canvas = cv2.addWeighted(frame_copia, 0.3, np.zeros_like(frame_copia), 0, 0)
    canvas[y1:y2, x1:x2] = anomalia_bgr

    return canvas, estado_limpieza
