import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from skimage.filters import threshold_multiotsu


#CONFIGURACIÓN REALSENSE

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Cámara lista. Presiona 'q' para salir.")


try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        h, w, _ = img.shape

        # RECORTE CENTRAL
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        crop_img = img[margin_h:h - margin_h, margin_w:w - margin_w]

        # MEJORA PARA UMBRALIZACIÓN

        gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)

        thresholds = threshold_multiotsu(gray_blur, classes=4)
        labels = np.digitize(gray_blur, bins=thresholds)
        otsu_layer = (labels * (255 // labels.max())).astype(np.uint8)

        # CONVERSIÓN A HSV

        hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)

        # RANGOS DE COLOR

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

        # MÁSCARAS 

        mask_red = cv.addWeighted(
            cv.inRange(hsv, lower_red1, upper_red1), 1.0,
            cv.inRange(hsv, lower_red2, upper_red2), 1.0, 0
        )

        mask_white = cv.inRange(hsv, lower_white, upper_white)
        mask_black = cv.inRange(hsv, lower_black, upper_black)

        final_layer = np.zeros(mask_red.shape, dtype=np.uint8)
        final_layer[mask_black > 0] = 85
        final_layer[mask_white > 0] = 170
        final_layer[mask_red > 0]   = 255

        #BOUNDING BOX

        img_box = crop_img.copy()

        def bounding_box(mask, color, etiqueta):
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv.contourArea(cnt) < 500:
                    continue
                x, y, w_box, h_box = cv.boundingRect(cnt)
                cv.rectangle(img_box, (x, y), (x + w_box, y + h_box), color, 2)
                cv.putText(img_box, etiqueta, (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        bounding_box(mask_black, (0, 0, 255), "Marcador")
        bounding_box(mask_white, (255, 255, 0), "Regla")
        bounding_box(mask_red,   (0, 255, 0), "Hoja")

        cv.imshow('RealSense - Original', img)
        cv.imshow('Recorte', crop_img)
        cv.imshow('Otsu Multi-Nivel', otsu_layer)
        cv.imshow('Capa Final Color', final_layer)
        cv.imshow('Bounding Boxes', img_box)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv.destroyAllWindows()
