#metricas de error en filtros, media, mediana, gaussiano, vs original
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread("./images/color5.jpg")
cv.imshow("Imagen ruido", img)


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

#########mediana
#MSE 
print("Métrica de error para filtro de mediana:")
mse= np.mean((img - img_mediana)**2)
print("MSE:", mse)

#RMSE
rmse= np.sqrt(mse)
print("RMSE:", rmse)

#PSNR
psnr= 20*np.log10(255/rmse)
print("PSNR:", psnr)


#########media
print("Métrica de error para filtro de media:")
#MSE 
mse= np.mean((img - img_media)**2)
print("MSE:", mse)

#RMSE
rmse= np.sqrt(mse)
print("RMSE:", rmse)

#PSNR
psnr= 20*np.log10(255/rmse)
print("PSNR:", psnr)

##########gaussiano
print("Métrica de error para filtro de gaussiano:")
#MSE 
mse= np.mean((img - img_gaussiana)**2)
print("MSE:", mse)

#RMSE
rmse= np.sqrt(mse)
print("RMSE:", rmse)

#PSNR
psnr= 20*np.log10(255/rmse)
print("PSNR:", psnr)
cv.waitKey(0)
cv.destroyAllWindows()