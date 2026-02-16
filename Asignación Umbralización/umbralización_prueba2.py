import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu

cap = cv.VideoCapture('1.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape

    #  RECORTE CENTRAL 
    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    crop_img = img[margin_h:h-margin_h, margin_w:w-margin_w]


    # QUITAR SOMBRAS (LAB) REVISAR

    lab = cv.cvtColor(crop_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    background = cv.GaussianBlur(l, (51, 51), 0)
    l_shadowless = cv.subtract(l, background)
    l_shadowless = cv.normalize(l_shadowless, None, 0, 255, cv.NORM_MINMAX)

    lab_clean = cv.merge((l_shadowless, a, b))
    img_no_shadow = cv.cvtColor(lab_clean, cv.COLOR_LAB2BGR)


    # REALCE DE COLOR (HSV)

    hsv_boost = cv.cvtColor(img_no_shadow, cv.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv.split(hsv_boost)

    s_ch = cv.multiply(s_ch, 1.4)
    s_ch = np.clip(s_ch, 0, 255).astype(np.uint8)

    hsv_boost = cv.merge((h_ch, s_ch, v_ch))
    img_color_boost = cv.cvtColor(hsv_boost, cv.COLOR_HSV2BGR)


    # REALCE DE BORDES (Unsharp Mask)

    blur = cv.GaussianBlur(img_color_boost, (5, 5), 0)
    sharpened = cv.addWeighted(img_color_boost, 1.5, blur, -0.5, 0)


    # IMAGEN FINAL PARA THRESHOLD

    gray = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    #  MULTI-OTSU 
    thresholds = threshold_multiotsu(gray, classes=3)
    labels = np.digitize(gray, bins=thresholds)
    otsu_layer = (labels * (255 // labels.max())).astype(np.uint8)


    # HSV PARA COLOR (USANDO IMAGEN MEJORADA)

    hsv = cv.cvtColor(sharpened, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    mask_red = cv.addWeighted(
        cv.inRange(hsv, lower_red1, upper_red1), 1.0,
        cv.inRange(hsv, lower_red2, upper_red2), 1.0, 0
    )
    mask_white = cv.inRange(hsv, lower_white, upper_white)
    mask_black = cv.inRange(hsv, lower_black, upper_black)


    # CAPA FINAL COMBINADA

    final_layer = np.zeros(mask_red.shape, dtype=np.uint8)
    final_layer[mask_black > 0] = 85
    final_layer[mask_white > 0] = 170
    final_layer[mask_red > 0]   = 255


    #  BOUNDING BOXES

    img_box = sharpened.copy()

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


    # VISUALIZACIÓN

    cv.imshow('Recorte Original', crop_img)
    cv.imshow('Sin Sombras + Color', sharpened)
    cv.imshow('Otsu Multi-Nivel', otsu_layer)
    cv.imshow('Capa Final Color', final_layer)
    cv.imshow('Bounding Boxes', img_box)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
