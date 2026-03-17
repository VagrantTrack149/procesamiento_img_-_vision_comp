import pyrealsense2 as rs
import numpy as np
import cv2 as cv

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Parámetros de la banda
BANDA_LEFT = 100
BANDA_TOP = 200
BANDA_RIGHT = 900
BANDA_BOTTOM = 500

VENTANA = 13  # 13x13, 9 o 21


try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Crear máscara del fondo (todo lo que no es banda)
        mascara_banda = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.rectangle(mascara_banda, (BANDA_LEFT, BANDA_TOP), (BANDA_RIGHT, BANDA_BOTTOM), 255, -1)
        mascara_fondo = cv.bitwise_not(mascara_banda)
        fondo_pixels = mascara_fondo == 255

        # Aplicar filtros solo al fondo
        # Promedio
        img_prom = cv.blur(frame, (VENTANA, VENTANA))
        # Gaussiano
        img_gauss = cv.GaussianBlur(frame, (VENTANA, VENTANA), 0)
        #mezcla
        img_pro_gauss=cv.GaussianBlur(img_prom,(VENTANA,VENTANA),0)
        # Mexclar imágenes original (banda) + filtrado (fondo)
        final_prom = frame.copy()
        final_gauss = frame.copy()
        final_pro_gauss=frame.copy()
        final_prom[fondo_pixels] = img_prom[fondo_pixels]
        final_gauss[fondo_pixels] = img_gauss[fondo_pixels]
        final_pro_gauss[fondo_pixels]=img_pro_gauss[fondo_pixels]
        # Canny sobre la imagen con fondo mezcla
        gris_gauss = cv.cvtColor(final_pro_gauss, cv.COLOR_BGR2GRAY)
        bordes_canny = cv.Canny(gris_gauss, 100, 200)

        # Sobel 
        sobel_x = cv.Sobel(gris_gauss, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(gris_gauss, cv.CV_64F, 0, 1, ksize=3)
        bordes_sobel = cv.convertScaleAbs(sobel_x + sobel_y)
        cv.imshow('Original', frame)
        cv.imshow('Fondo Promedio', final_prom)
        cv.imshow('Fondo Gaussiano', final_gauss)
        cv.imshow('Fondo pro+gauss',img_pro_gauss)
        cv.imshow('Bordes Canny', bordes_canny)
        cv.imshow('Bordes Sobel',bordes_sobel)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv.destroyAllWindows()