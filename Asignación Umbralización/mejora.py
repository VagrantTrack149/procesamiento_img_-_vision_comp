#mejorar el primer frame para la segmentación y umbralización
import cv2 as cv
import numpy as np


img = cv.imread('primer_fotograma.jpg')
h, w, _ = img.shape #temporales de tamaño de la imagen para recorte

# Ajustamos un margen del 20% en los bordes(recortamos el centro para evitar problemas de fondo)
margin_h, margin_w = int(h * 0.2), int(w * 0.2)
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

final_layer = np.zeros(mask_red.shape, dtype=np.uint8)

final_layer[mask_black > 0] = 85 # Capa 1: Marcador 
final_layer[mask_white > 0] = 170 # Capa 2: Regla 
final_layer[mask_red > 0] = 255 # Capa 3: Hoja

cv.imshow('Recorte Original', crop_img)
cv.imshow('Capa Final', final_layer)
cv.imshow('Capa 1 - Marcador Negro', layer_black)
cv.imshow('Capa 2 - Regla Blanca', layer_white)
cv.imshow('Capa 3 - Hoja Roja', layer_red)

#Bounding box
img_box=img.copy()
def bbounding_box(mask, color,etiqueta):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        x+= margin_w
        y+= margin_h
        cv.rectangle(img_box, (x, y), (x+w, y+h), color, 2)
        cv.putText(img_box, etiqueta, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

bbounding_box(mask_black, (0, 0, 255), "Marcador")
bbounding_box(mask_white, (255, 255, 0), "Regla")
bbounding_box(mask_red, (0, 255, 0), "Hoja")

cv.imshow('Bounding Boxes', img_box)

cv.waitKey(0)
cv.destroyAllWindows()