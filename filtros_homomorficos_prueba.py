import cv2 as cv
import numpy as np

img = cv.imread("./images/tunel.jpeg")
if img is None:
	raise FileNotFoundError("Image not found: ./images/tunel.jpeg")
cv.imshow("Imagen Original", img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)
h, w = img_gray.shape

# Homomorphic filter
# use log1p to avoid log(0) warnings
I1 = np.log1p(img_gray)
FI = np.fft.fft2(I1)
FI_shift = np.fft.fftshift(FI)

gh = 0.4
gl = 0.2
Do = 4
c = -0.5

# build distance grid matching image shape (y,x) so shapes align with FFT arrays
y, x = np.indices((h, w))
yc = y - (h // 2)
xc = x - (w // 2)
D = (xc**2 + yc**2)
H = (gh - gl) * (1 - np.exp(c * (D / (Do**2)))) + gl

# apply filter in frequency domain with proper shifting
G = H * FI_shift
G_ishift = np.fft.ifftshift(G)
i2 = np.fft.ifft2(G_ishift)
i3 = np.real(np.exp(i2)) - 1

i3 = cv.normalize(i3, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
cv.imshow("Imagen Mejorada", i3)

cv.waitKey(0)
cv.destroyAllWindows()