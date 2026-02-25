import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu

cap = cv.VideoCapture('1.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape

    # Región de interés centrada
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    crop_img = img[margin_h:h-margin_h, margin_w:w-margin_w]

    # quitar sombras
    #https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
    lab = cv.cvtColor(crop_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    background = cv.GaussianBlur(l, (51, 51), 0)
    l_shadowless = cv.subtract(l, background)
    l_shadowless = cv.normalize(l_shadowless, None, 0, 255, cv.NORM_MINMAX)

    lab_clean = cv.merge((l_shadowless, a, b))
    img_no_shadow = cv.cvtColor(lab_clean, cv.COLOR_LAB2BGR)

    # Incrementar saturación
    #https://es.stackoverflow.com/questions/226799/c%C3%B3mo-modificar-intensidades-de-colores-en-una-imagen-con-numpy-python

    hsv = cv.cvtColor(img_no_shadow, cv.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv.split(hsv)
    s_ch = cv.multiply(s_ch, 1.4)
    s_ch = np.clip(s_ch, 0, 255).astype(np.uint8)
    hsv_boost = cv.merge((h_ch, s_ch, v_ch))
    img_color_boost = cv.cvtColor(hsv_boost, cv.COLOR_HSV2BGR)

    # filtro sharpen
    #https://pytutorial.com/python-basic-image-sharpening-methods/
    blur = cv.GaussianBlur(img_color_boost, (5, 5), 0)
    sharpened = cv.addWeighted(img_color_boost, 1.5, blur, -0.5, 0)

    # Máscara de color para fondo 
    # Eliminar fondo de color picker
    lab2 = cv.cvtColor(sharpened, cv.COLOR_BGR2LAB)
    bg_mask = cv.inRange(
        lab2,
        np.array([0, 120, 120]),   # fondo 
        np.array([255, 136, 136])
    )
    fg_mask_color = cv.bitwise_not(bg_mask)

    # MULTI-OTSU con preprocesado gauss escala grises
    gray = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    thresholds = threshold_multiotsu(gray, classes=4)
    labels = np.digitize(gray, bins=thresholds)

    otsu_mask = np.zeros_like(gray, dtype=np.uint8)
    otsu_mask[(labels == 1) | (labels == 2)] = 255

    # INCLUIR OBJETOS NEGROS (GRADIENTE)
    # https://www.geeksforgeeks.org/python/python-morphological-operations-in-image-processing-gradient-set-3/ 

    grad = cv.morphologyEx(gray, cv.MORPH_GRADIENT,
                           cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
    _, grad_mask = cv.threshold(grad, 15, 255, cv.THRESH_BINARY)

    # Unir mascaras
    object_mask = cv.bitwise_and(fg_mask_color, otsu_mask)
    object_mask = cv.bitwise_or(object_mask, grad_mask)

    ## Rellenar huecos en las areas
    #https://stackoverflow.com/questions/47517667/how-does-cv2-floodfill-work

    h_mask, w_mask = object_mask.shape
    floodfill = object_mask.copy()
    mask = np.zeros((h_mask + 2, w_mask + 2), np.uint8)
    cv.floodFill(floodfill, mask, (0, 0), 255)
    floodfill_inv = cv.bitwise_not(floodfill)
    object_mask = cv.bitwise_or(object_mask, floodfill_inv)

    # Morfologia
    # https://programmerclick.com/article/44381457787/ 
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) 
    object_mask = cv.morphologyEx(object_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    object_mask = cv.morphologyEx(object_mask, cv.MORPH_OPEN, kernel, iterations=1)

    img_box = sharpened.copy()
    num_labels, labels_cc, stats, centroids = cv.connectedComponentsWithStats(object_mask, connectivity=8)

    # BOUNDING BOX generico copiando los bordes
    obj_id = 1
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area < 1200:
            continue
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w_box = stats[i, cv.CC_STAT_WIDTH]
        h_box = stats[i, cv.CC_STAT_HEIGHT]

        cv.rectangle(img_box, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv.putText(img_box, f"Objeto {obj_id}",
                   (x, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0), 2)
        obj_id += 1



    # Mostrar resultados
    cv.imshow('multi-otsu', otsu_mask)
    cv.imshow("Recorte", crop_img)
    cv.imshow("Labels",labels.astype(np.uint8)*85)
    cv.imshow("Sin Fondo + Objetos", object_mask)
    cv.imshow("Bounding Boxes", img_box)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()