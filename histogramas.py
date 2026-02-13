#histogramas
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img= cv.imread("./images/tunel.jpeg")
cv.imshow("Imagen Original", img)

#histograma de una imagen
histogram= cv.calcHist([img], [0], None, [256], [0,256])
plt.figure(1)
plt.plot(histogram)
plt.show()

#histograma a color
#colors= ('b','g','r')
#for i, color in enumerate(colors):
#    histogram= cv.calcHist([img], [i], None, [256], [0,256])
#    plt.plot(histogram, color=color)
#    plt.xlim([0,256])
#plt.show()

#histograma ecualizado blanco y negro
img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
histogram= cv.calcHist([img_gray], [0], None, [256], [0,256])
plt.figure(2)
plt.plot(histogram)
plt.show()

img_eq= cv.equalizeHist(img_gray)
histogram_eq= cv.calcHist([img_eq], [0], None, [256], [0,256])
plt.figure(3)
plt.plot(histogram_eq)
plt.show()
cv.imshow("Imagen Ecualizada", img_eq)

#histograma ecualizado a color
img_eq_color= cv.cvtColor(img, cv.COLOR_BGR2HSV)
img_eq_color[:,:,2]= cv.equalizeHist(img_eq_color[:,:,2])
img_eq_color= cv.cvtColor(img_eq_color, cv.COLOR_HSV2BGR)
cv.imshow("Imagen Ecualizada a Color", img_eq_color)

#imagen gamma funcion
def gamma_correction(image, gamma):
    invGamma= 1/gamma
    table= [((i/255.0) ** invGamma) * 255 for i in range(256)]
    table= np.array(table, np.uint8)
    return cv.LUT(image, table)
img_gamma_corrected= gamma_correction(img, 1.2)
cv.imshow("Imagen con Correccion Gamma", img_gamma_corrected)

#clahe
clahe= cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe= clahe.apply(img_gray)
cv.imshow("Imagen con CLAHE", img_clahe)


cv.waitKey(0)
cv.destroyAllWindows()