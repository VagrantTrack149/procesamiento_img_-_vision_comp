#filtro pasa altas bueno
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread("./images/tablero.jpg")
cv.imshow("Imagen Original", img)

#filtro promedio
kernel= np.ones((5,5), np.float32)/25
img_media= cv.filter2D(img, -1, kernel)
cv.imshow("Imagen Media", img_media)

#filtro bordes
kernel= np.array([[-1,-1,-1],
                  [-1, 8,-1],
                  [-1,-1,-1]])
img_bordes= cv.filter2D(img, -1, kernel)
img_bordes[img_bordes > 30] = 255

cv.imshow("Imagen Bordes", img_bordes)




cv.waitKey(0)
cv.destroyAllWindows()