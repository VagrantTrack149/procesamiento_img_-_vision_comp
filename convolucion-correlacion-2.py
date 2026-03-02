import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
img = cv.imread('images/image100.png', cv.IMREAD_UNCHANGED)
img_ob=cv.imread('images/image.png', cv.IMREAD_UNCHANGED)

cv.imshow("Imagen Original", img)

# Recortar template
#imf = img[120:120+650, 30:30+250]
imf=img[200:400, 400:600]
plt.figure(1)
if img.ndim == 2:
    plt.imshow(img, cmap='gray')
else:
    plt.imshow(cv.cvtColor(imf, cv.COLOR_BGR2RGB))
plt.title('Imagen Original')
plt.axis('off')
plt.show()
# Recortar cd 
imT = imf[90:190,97:156]
#gamma correction a imT
def gamma_correction(image, gamma):
    invGamma= 1/gamma
    table= [((i/255.0) ** invGamma) * 255 for i in range(256)]
    table= np.array(table, np.uint8)
    return cv.LUT(image, table)
imT=gamma_correction(imT, 1.2)

#imf=img_ob.copy()

print(f"Imagen template: {imf.shape}")
print(f"cd: {imT.shape}")

cv.imshow("Imagen Template", imf)
cv.imshow("cd", imT)
h1, w1 = imf.shape[:2]
h2, w2 = imT.shape[:2]

# Tamaño de padding 
pad_h = h1 + h2
pad_w = w1 + w2

# Convertir a escala de grises
if imf.ndim == 3:
    imf_gray = cv.cvtColor(imf, cv.COLOR_BGR2GRAY)
else:
    imf_gray = imf

if imT.ndim == 3:
    imT_gray = cv.cvtColor(imT, cv.COLOR_BGR2GRAY)
else:
    imT_gray = imT

# Hacer padding a cero
imf_pad = np.zeros((pad_h, pad_w), dtype=np.float32)
imf_pad[:h1, :w1] = imf_gray.astype(np.float32)

imT_pad = np.zeros((pad_h, pad_w), dtype=np.float32)
imT_pad[:h2, :w2] = imT_gray.astype(np.float32)

# Transformada de Fourier
f_imf = np.fft.fft2(imf_pad)
fshift_imf = np.fft.fftshift(f_imf)

f_imT = np.fft.fft2(imT_pad)
fshift_imT = np.fft.fftshift(f_imT)

# Correlación: F(imf) * conj(F(imT))
correlacion_freq = fshift_imf * np.conj(fshift_imT)

# Transformada inversa
result_freq = np.fft.ifftshift(correlacion_freq)
correlacion_spatial = np.fft.ifft2(result_freq)
correlacion_spatial = np.abs(correlacion_spatial)

# Normalizar para visualización
correlacion_norm = cv.normalize(correlacion_spatial, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Mostrar resultado
#plt.figure(2)
#plt.imshow(correlacion_norm, cmap='hot')
#plt.title('Mapa de Correlación (FFT)')
#plt.colorbar()
#plt.show()

max_val=np.max(correlacion_norm)
print(max_val)
cv.imshow("Resultado Correlación", correlacion_norm)

#correlacion max
threshold = 0.97 * max_val
# creamos una máscara binaria basada en ese umbral
coor_max = np.where(correlacion_norm >= threshold, 255, 0).astype(np.uint8)
cv.imshow("Resultado Correlación 95%", coor_max)
[y, x]=np.where(coor_max==255)
print(x)
print(y)
print('posicion del maximo')
#colocar parche del cd(imT) en la posición y,x
#coor_max[y/2:y/2 +h2,x/2:x/2 +w2]=imT

cv.waitKey(0)
cv.destroyAllWindows()
