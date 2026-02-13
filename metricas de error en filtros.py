#metricas de error en filtros ruidosa vs original
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_ruido= cv.imread("./images/color5.jpg")
cv.imshow("Imagen ruido", img_ruido)
img= cv.imread("./images/color5original.png")
cv.imshow("Imagen Original", img)

#MSE 
mse= np.mean((img - img_ruido)**2)
print("MSE:", mse)

#RMSE
rmse= np.sqrt(mse)
print("RMSE:", rmse)

#PSNR
psnr= 20*np.log10(255/rmse)
print("PSNR:", psnr)




cv.waitKey(0)
cv.destroyAllWindows()