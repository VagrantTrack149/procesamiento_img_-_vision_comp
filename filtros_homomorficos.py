import cv2 as cv
import numpy as np
import time 
img = cv.imread("./images/tunel.jpeg")
cv.imshow("Imagen Original", img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)
h, w = img_gray.shape
t0=time.time()
#filtro homomorfico
I1 = np.log1p(img_gray)
FI = np.fft.fftshift(np.fft.fft2(I1))

gh = 0.4
gl = 0.1
Do = 4
c = -0.5
u = np.arange(h) - h // 2
v = np.arange(w) - w // 2
V, U = np.meshgrid(v, u)

D = (U**2 + V**2)
H = (gh - gl) * (1 - np.exp(c * (D / (Do**2)))) + gl

G = H * FI
i2 = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
i3 = np.exp(i2) - 1

i3 = cv.normalize(i3, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
t1=time.time()
print('tiempo transcurrido en s:'+ str(t1-t0))

cv.imshow("Imagen Mejorada", i3)

cv.waitKey(0)
cv.destroyAllWindows()