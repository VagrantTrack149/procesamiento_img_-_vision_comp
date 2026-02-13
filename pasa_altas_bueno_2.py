#filtro espcial sharpeining
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters
img= cv.imread("./images/tablero.jpg")
cv.imshow("Imagen Original", img)

#filtro de sharpening
lap = cv.Laplacian(img, cv.CV_64F, ksize=3)
lap_vis=cv.convertScaleAbs(lap)

lap_vis_gris= cv.cvtColor(lap_vis, cv.COLOR_BGR2GRAY)
cv.imshow("Imagen laplaciano", lap_vis_gris)

#filtro bordes
kernel= np.array([[-1,-1,-1],
                  [-1, 8,-1],
                  [-1,-1,-1]])
img_bordes= cv.filter2D(img, -1, kernel)
img_bordes[img_bordes > 30] = 255
#convertir a escala de grises
img_bordes_gray= cv.cvtColor(img_bordes, cv.COLOR_BGR2GRAY)
cv.imshow("Imagen bordes", img_bordes_gray)

#filtro laplaciano con mascara gaussiana
lap_gauss= cv.GaussianBlur(img, (3,3), 0)
lap_gauss= cv.Laplacian(lap_gauss, cv.CV_64F, ksize=3)
lap_gauss_vis=cv.convertScaleAbs(lap_gauss)
lap_gauss_vis_gris= cv.cvtColor(lap_gauss_vis, cv.COLOR_BGR2GRAY)
cv.imshow("Imagen laplaciano con mascara gaussiana", lap_gauss_vis_gris)


#filtro sobel
sobelx= cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely= cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
sobel_vis_x= cv.convertScaleAbs(sobelx)
sobel_vis_y= cv.convertScaleAbs(sobely)
sobel_vis= cv.addWeighted(sobel_vis_x, 0.5, sobel_vis_y, 0.5, 0)
sobel_vis_gris= cv.cvtColor(sobel_vis, cv.COLOR_BGR2GRAY)
cv.imshow("Imagen Sobel imagen completa", sobel_vis_gris)

#filtro prewitt
kernelx= np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernely= np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
prewittx= cv.filter2D(img, -1, kernelx)
prewitty= cv.filter2D(img, -1, kernely)
prewitt_vis_x= cv.convertScaleAbs(prewittx)
prewitt_vis_y= cv.convertScaleAbs(prewitty)
prewitt_vis=cv.addWeighted(prewitt_vis_x, 0.5, prewitt_vis_y, 0.5, 0)
prewitt_vis_gris= cv.cvtColor(prewitt_vis, cv.COLOR_BGR2GRAY)
cv.imshow("Imagen Prewitt imagen completa", prewitt_vis_gris)

#filtrado canny
img_canny= cv.Canny(img, 100, 200)
cv.imshow("Imagen Canny", img_canny)

#filtraod roberts
#roberts_cross_v = np.array( [[1, 0 ],
#                             [0,-1 ]] )

#roberts_cross_h = np.array( [[ 0, 1 ],
#                             [ -1, 0 ]] )

#img_roberts= img.astype('float64')
#img_roberts /= 255.0
#vertical = ndimage.convolve( img_roberts, roberts_cross_v )
#horizontal = ndimage.convolve( img_roberts, roberts_cross_h )

#edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
#edged_img*=255
#cv.imshow("Imagen Roberts", edged_img)



#filtrado gradiente
sobelx= cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely= cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
gradient_magnitude= np.sqrt(sobelx**2 + sobely**2)
gradient_magnitude= cv.convertScaleAbs(gradient_magnitude)
gradient_magnitude_gray= cv.cvtColor(gradient_magnitude, cv.COLOR_BGR2GRAY)
gradient_magnitude_gray[gradient_magnitude_gray > 100] = 255
cv.imshow("Imagen Gradiente", gradient_magnitude_gray)

#filtro gradienete magnitud
sobelx= cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely= cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
gradient_magnitude= np.sqrt(sobelx**2 + sobely**2)
gradient_magnitude= cv.convertScaleAbs(gradient_magnitude)
gradient_magnitude_gray= cv.cvtColor(gradient_magnitude, cv.COLOR_BGR2GRAY)
cv.imshow("Imagen Gradiente Magnitud", gradient_magnitude_gray)
#filtro gradiente orientacion
sobelx= cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely= cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
gradient_orientation= np.arctan2(sobely, sobelx)
gradient_orientation_vis= cv.convertScaleAbs(gradient_orientation)

gradient_orientation_vis_gray= cv.cvtColor(gradient_orientation_vis, cv.COLOR_BGR2GRAY)

cv.imshow("Imagen Gradiente Orientación", gradient_orientation_vis_gray)



cv.waitKey(0)
cv.destroyAllWindows()