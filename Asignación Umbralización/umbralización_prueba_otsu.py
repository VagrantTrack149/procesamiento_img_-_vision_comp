import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu

cap = cv.VideoCapture('1.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape

    # Ajustamos región de interes al centro

    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    crop_img = img[margin_h:h-margin_h, margin_w:w-margin_w]

    # quitar sombras
    lab = cv.cvtColor(crop_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    background = cv.GaussianBlur(l, (51, 51), 0)
    l_shadowless = cv.subtract(l, background)
    l_shadowless = cv.normalize(l_shadowless, None, 0, 255, cv.NORM_MINMAX)

    lab_clean = cv.merge((l_shadowless, a, b))
    img_no_shadow = cv.cvtColor(lab_clean, cv.COLOR_LAB2BGR)

    # Incrementar saturación
    hsv = cv.cvtColor(img_no_shadow, cv.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv.split(hsv)

    s_ch = cv.multiply(s_ch, 1.4)
    s_ch = np.clip(s_ch, 0, 255).astype(np.uint8)

    hsv_boost = cv.merge((h_ch, s_ch, v_ch))
    img_color_boost = cv.cvtColor(hsv_boost, cv.COLOR_HSV2BGR)

    # filtro sharpen
    blur = cv.GaussianBlur(img_color_boost, (5, 5), 0)
    sharpened = cv.addWeighted(img_color_boost, 1.5, blur, -0.5, 0)

    # eliminar fondo de color picker
    lab2 = cv.cvtColor(sharpened, cv.COLOR_BGR2LAB)

    bg_mask = cv.inRange(
        lab2,
        np.array([0, 120, 120]),   # fondo gris verdoso
        np.array([255, 136, 136])
    )

    fg_mask_color = cv.bitwise_not(bg_mask)

    #  MULTI-OTSU con procesado gauss
    gray = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    thresholds = threshold_multiotsu(gray, classes=3)
    labels = np.digitize(gray, bins=thresholds)

    otsu_mask = np.zeros_like(gray, dtype=np.uint8)
    otsu_mask[(labels == 1) | (labels == 2)] = 255

    # Unir mascaras
    object_mask = cv.bitwise_and(fg_mask_color, otsu_mask)
    #rellenar huevos en areas
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    object_mask = cv.morphologyEx(object_mask, cv.MORPH_CLOSE, kernel)
    object_mask = cv.morphologyEx(object_mask, cv.MORPH_OPEN, kernel)

    #bounding box generico copiando los bordes
    img_box = sharpened.copy()
    contours, _ = cv.findContours(object_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 800:
            continue

        x, y, w_box, h_box = cv.boundingRect(cnt)
        cv.rectangle(img_box, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv.putText(img_box, "Objeto",
                   (x, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0), 2)

    cv.imshow("Recorte", crop_img)
    cv.imshow("Sin Fondo + Objetos", object_mask)
    cv.imshow("Bounding Boxes", img_box)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
