import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os


# Configuración de la cámara

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciar el pipeline
pipeline.start(config)

print("Cámara lista. Presiona 'q' para salir.")


# Loop principal

try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())

        h, w, _ = img.shape  # temporales de tamaño de la imagen para recorte

        # Ajustamos un margen del 20% en los bordes(recortamos el centro para evitar problemas de fondo)
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        crop_img = img[margin_h:h-margin_h, margin_w:w-margin_w]

        # Convertir a HSV para mejor detección de color
        hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)

        
        # Definir rangos de colores
        
        # Rojo
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Blanco
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])

        # Negro
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])

        
        # Máscaras
        
        mask_red = cv.addWeighted(
            cv.inRange(hsv, lower_red1, upper_red1), 1.0,
            cv.inRange(hsv, lower_red2, upper_red2), 1.0, 0
        )

        mask_white = cv.inRange(hsv, lower_white, upper_white)
        mask_black = cv.inRange(hsv, lower_black, upper_black)

        
        # Capas binarias
        
        layer_black = np.zeros_like(mask_black)
        layer_white = np.zeros_like(mask_white)
        layer_red   = np.zeros_like(mask_red)

        layer_black[mask_black > 0] = 255
        layer_white[mask_white > 0] = 255
        layer_red[mask_red > 0]     = 255

        
        # Bounding box
        
        img_box = crop_img.copy()

        def bbounding_box(mask, color, etiqueta):
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv.contourArea(cnt) < 500:  # filtrar ruido
                    continue
                x, y, w_box, h_box = cv.boundingRect(cnt)
                #x += margin_w
                #y += margin_h
                cv.rectangle(img_box, (x, y), (x + w_box, y + h_box), color, 2)
                cv.putText(img_box, etiqueta, (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        bbounding_box(mask_black, (0, 0, 255), "Marcador")
        bbounding_box(mask_white, (255, 255, 0), "Regla")
        bbounding_box(mask_red,   (0, 255, 0), "Hoja")

        
        
        cv.imshow('RealSense - Original', img)
        cv.imshow('Recorte', crop_img)
        cv.imshow('Capa 1 - Marcador Negro', layer_black)
        cv.imshow('Capa 2 - Regla Blanca', layer_white)
        cv.imshow('Capa 3 - Hoja Roja', layer_red)
        cv.imshow('Bounding Boxes', img_box)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv.destroyAllWindows()
