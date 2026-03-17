import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from skimage.filters import threshold_multiotsu

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# ROI donde se espera el código de barras
ROI_TOP = 220
ROI_BOTTOM = 450
ROI_LEFT = 100
ROI_RIGHT = 900

# Tamaño de ventana para filtros
VENTANA = 13  # 9, 13 o 21

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Extraer ROI
        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        roi_gris = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # Segmentación multi-Otsu para aislar el código
        thresholds = threshold_multiotsu(roi_gris, classes=4)
        labels = np.digitize(roi_gris, bins=thresholds)
        mascara = np.zeros(roi_gris.shape, dtype=np.uint8)

        # Etiquetas 1 y 3
        mascara[(labels == 1) | (labels == 3)] = 255
        codigo_pixels = mascara == 255

        # Aplicar filtros al ROI
        roi_prom = cv.blur(roi, (VENTANA, VENTANA))
        roi_gauss = cv.GaussianBlur(roi, (VENTANA, VENTANA), 0)
        roi_gauss_prom=cv.GaussianBlur(roi_gauss,(VENTANA,VENTANA),0)
        # Crear versiones difu
        roi_anon_prom = roi.copy()
        roi_anon_gauss = roi.copy()
        roi_anon_gauss_prom=roi.copy()
        roi_anon_prom[codigo_pixels] = roi_prom[codigo_pixels]
        roi_anon_gauss[codigo_pixels] = roi_gauss[codigo_pixels]
        roi_anon_gauss_prom[codigo_pixels]=roi_gauss_prom[codigo_pixels]

        # Insertar el ROI modificado de vuelta al frame completo
        frame_prom = frame.copy()
        frame_gauss = frame.copy()
        frame_gauss_prom=frame.copy()
        frame_prom[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = roi_anon_prom
        frame_gauss[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = roi_anon_gauss
        frame_gauss_prom[ROI_TOP:ROI_BOTTOM,ROI_LEFT:ROI_RIGHT]=roi_anon_gauss_prom

        cv.imshow('Original', frame)
        cv.imshow('Codigo Promedio', frame_prom)
        cv.imshow('Codigo Gaussiano', frame_gauss)
        cv.imshow('Codigo difu gauss+ prom',frame_gauss_prom)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv.destroyAllWindows()