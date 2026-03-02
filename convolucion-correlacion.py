import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread(r'images\image2.tif', cv.IMREAD_UNCHANGED)
# graficando img con matplotlib (usar imshow en lugar de plot)
plt.figure(1)
if img is None:
	raise FileNotFoundError(r"images\image2.tif not found")
if img.ndim == 2:
	plt.imshow(img, cmap='gray')
else:
	plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Imagen')
plt.axis('off')
plt.show()

cv.imshow("Imagen Original", img)
#print(img.shape)

# Recortar sección "cd"
# Altura (Y) de 90 a 160 | Ancho (X) de 250 a 650
imf = img[130:770, 48:250] 

# Recortar "template"
# Altura (Y) de 48 a 99 | Ancho (X) de 35 a 60
imT = img[150:220, 99:200]

#recortar cd
print(imf.shape)

print(imT.shape)

cv.imshow("Imagen tempalte", imf)
cv.imshow("Imagen cd", imT)   

#convertimos a 0 todos los valores menos los del cd en una variable temporal del template
imgT_temp=imT.copy()
plt.plot(imgT_temp)

#transformada de fouier
f = np.fft.fft2(imf)
fshift = np.fft.fftshift(f)

f1=np.fft.fft2(imT)
fshift1 = np.fft.fftshift(f1)

#calculo de conjugado del cd

#correlación frecuencial transformada de fourier con 
#img_final=

#transformada inversa de fourier con la img_final


cv.waitKey(0)
cv.destroyAllWindows()