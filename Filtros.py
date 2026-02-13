#filtros espaciales
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread("./images/color5.jpg")
cv.imshow("Imagen Original", img)

#filtro de media
kernel= np.ones((5,5), np.float32)/25
img_media= cv.filter2D(img, -1, kernel)
cv.imshow("Imagen Media", img_media)

#filtro de mediana
img_mediana= cv.medianBlur(img, 11) #ventana de 11 x 11
cv.imshow("Imagen Mediana", img_mediana)

#filtro de gausiano
img_gaussiana= cv.GaussianBlur(img, (11,11), 1) #ventana de 11 x 11 y sigma(desviación estandar)
cv.imshow("Imagen Gaussiana", img_gaussiana)

#filtro de bilateral
img_bilateral= cv.bilateralFilter(img, 9, 75, 75)
cv.imshow("Imagen Bilateral", img_bilateral)

#filtro de laplaciano
kernel_laplaciano= np.array([[0,1,0],[1,-4,1],[0,1,0]])
img_laplaciana= cv.filter2D(img, -1, kernel_laplaciano)
cv.imshow("Imagen Laplaciana", img_laplaciana)

#filtro de sobel
kernel_sobel_x= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_sobel_y= np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
img_sobel_x= cv.filter2D(img, -1, kernel_sobel_x)
img_sobel_y= cv.filter2D(img, -1, kernel_sobel_y)
img_sobel= cv.magnitude(img_sobel_x.astype(np.float32), img_sobel_y.astype(np.float32))
cv.imshow("Imagen Sobel X", img_sobel_x)
cv.imshow("Imagen Sobel Y", img_sobel_y)
cv.imshow("Imagen Sobel", img_sobel)

#filtro de canny
img_canny= cv.Canny(img, 100, 200)
cv.imshow("Imagen Canny", img_canny)

#filtro de prewitt
kernel_prewitt_x= np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernel_prewitt_y= np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
img_prewitt_x= cv.filter2D(img, -1, kernel_prewitt_x)
img_prewitt_y= cv.filter2D(img, -1, kernel_prewitt_y)
img_prewitt= cv.magnitude(img_prewitt_x.astype(np.float32), img_prewitt_y.astype(np.float32))
cv.imshow("Imagen Prewitt X", img_prewitt_x)
cv.imshow("Imagen Prewitt Y", img_prewitt_y)
cv.imshow("Imagen Prewitt", img_prewitt)

cv.waitKey(0)
cv.destroyAllWindows()