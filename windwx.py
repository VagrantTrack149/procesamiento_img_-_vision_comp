import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread('images/tablero.jpg')
img_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#noramalizar la imagen
img_gris_norm = cv.normalize(img_gris, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
#filtro frecuencial pasa bajas
def aplicar_filtro_frecuencial(img_gris, radio):
    dft = cv.dft(np.float32(img_gris), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img_gris.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv.circle(mask, (ccol, crow), radio, (1, 1, 1), -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

res_final = aplicar_filtro_frecuencial(img_gris_norm, 30)
#mostrar espectro y mostrarlo como imagen
f = np.fft.fft2(img_gris)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.figure(1)
plt.imshow(magnitude_spectrum)
plt.title('Espectro de Frecuencia')
plt.colorbar()
plt.show()


#filtro gaussiano
img_gauss = cv.GaussianBlur(res_final, (9, 9), 35)
cv.imshow("Imagen Gaussiana", img_gauss)

#filtro pasa altas
#1-gaussiana
res_final_altas = cv.subtract(res_final, img_gauss)
res_final_altas = np.where(res_final_altas > 8, 255, 0).astype(np.uint8)
cv.imshow("Imagen Pasa Altas", res_final_altas)


cv.imshow("Imagen Original", img_gris)
cv.imshow("Imagen Filtrada", res_final)
cv.waitKey(0)
cv.destroyAllWindows()