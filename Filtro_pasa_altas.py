#filtro espacial sharpening
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from PIL import Image


img= cv.imread("./images/color5.jpg")
cv.imshow("Imagen Original", img)


#filtro de sharpening
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv.filter2D(img, -1, kernel)
cv.imshow("Imagen Sharpened", sharpened)

#filtro maximo

kernel= np.ones((5,5), np.float32)/25
img_media= cv.filter2D(img, -1, kernel)
img_array = np.array(img)
rango=1
maximo= signal.maximum_filter(img_array, rango)
resultado= Image.fromarray(maximo.astype('uint8'))
cv.imshow("Imagen Maximo", maximo)


#filtro extremo(order filter)
img_array = np.array(img)
domain= np.ones((5,5))
rango=1
extremo= signal.order_filter(img_array, domain, rango) #filtro de mínimo
resultado= Image.fromarray(extremo.astype('uint8'))
cv.imshow("Imagen Extremo", extremo)

cv.waitKey(0)
cv.destroyAllWindows()