#umbralización
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_multiotsu

img= cv.imread('./images/foto1.png', cv.IMREAD_GRAYSCALE)
cv.imshow("Imagen Original", img)

#umbralizacion binaria
ret, img_binaria= cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv.imshow("Imagen Binaria", img_binaria)

#umbralizacion adaptativa
img_adaptativa= cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
cv.imshow("Imagen Adaptativa", img_adaptativa)

#umbralizacion otsu
ret, img_otsu= cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Imagen Otsu", img_otsu)

#otsu real
#aplicarle una función GaussianBlur a la imagen original y luego aplicarle el método de otsu para obtener mejores resultados
img_gaussiana= cv.GaussianBlur(img, (5,5), 0)
cv.imshow("Imagen Gaussiana", img_gaussiana)
img_adaptativa= threshold_multiotsu(img_gaussiana, classes=5)
labels= np.digitize(img_gaussiana, bins=img_adaptativa)
cv.imshow("Imagen Otsu Real", labels.astype(np.uint8)*85)
capas=labels.astype(np.uint8)*85



cv.waitKey(0)
cv.destroyAllWindows()