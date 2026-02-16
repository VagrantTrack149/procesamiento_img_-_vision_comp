import cv2 as cv
import numpy as np

img = np.zeros((60, 60), dtype=np.float32)
img[18:42, 23:37] = 127

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#filtrado pasa bajas gausiano
def gaussian_filter(shape, sigma):
    m, n = shape
    y, x = np.ogrid[-m//2 : m//2, -n//2 : n//2]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h /= h.sum()
    return h

h_mask = gaussian_filter((60, 60), 8)

f_filtered = fshift * h_mask

f_ishift = np.fft.ifftshift(f_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

#filtrado pasa altas (gausiano inverso)
h_mask_inv = 1 - h_mask
f_filtered_altas = fshift * h_mask_inv

f_ishift_altas = np.fft.ifftshift(f_filtered_altas)
img_alta = np.fft.ifft2(f_ishift_altas)
img_alta = np.abs(img_alta)

magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

cv.imshow('Imagen Original', (img).astype(np.uint8))
cv.imshow('Mascara Gaussiana (Filtro)', h_mask / h_mask.max()) # Normalizado para ver
cv.imshow('Mascara Inversa', h_mask_inv / h_mask_inv.max()) # Normalizado para ver
cv.imshow('Resultado Pasa Bajas', cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8))
cv.imshow('Resultado Pasa Altas', cv.normalize(img_alta, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()