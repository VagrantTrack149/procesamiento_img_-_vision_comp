import cv2 as cv
import numpy as np
import time
from skimage.filters import threshold_multiotsu

def filtro_frecuencial_pasabajas(img_gris, radio):
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

# Configuración
video_path = 'videos/videos renombrar/2_codigo_amarillol.mp4'
ventanas = [9, 13, 21]
K_MAX = 47 

for n in ventanas:
    print(f"\nAnalizando Ventana {n}x{n} para Código de Barras")
    video = cv.VideoCapture(video_path)
    mse_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    tiempos = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    
    ret, frame_ref = video.read()
    if ret:
        roi_ref = cv.cvtColor(frame_ref[220:450, 100:900], cv.COLOR_BGR2GRAY)
        ref_blur = cv.blur(roi_ref, (K_MAX, K_MAX))
        mse_max = np.mean((roi_ref - ref_blur) ** 2)
    
    video.set(cv.CAP_PROP_POS_FRAMES, 0)

    while video.isOpened():
        ret, frame = video.read()
        if not ret: break
        
        # Segmentación del código de barras (ROI específica)
        img_cortada = cv.cvtColor(frame[220:450, 100:900], cv.COLOR_BGR2GRAY)
        
        # Multi-Otsu para localizar el código, como es blanco se utiliza label 3
        thresholds = threshold_multiotsu(img_cortada, classes=4)
        labels = np.digitize(img_cortada, bins=thresholds)
        
        # mascara fusionada para código de barras (label 3) y amarillo (label 2)
        mascara_roi = np.zeros(img_cortada.shape, dtype=np.uint8)
        mascara_roi[(labels == 3) | (labels == 2)] = 255
        
        # 1. Promedio
        t0 = time.time()
        res_prom = cv.blur(frame[220:450, 100:900], (n, n))
        tiempos['Promedio'].append(time.time() - t0)
        mse_total['Promedio'].append(np.mean((frame[220:450, 100:900] - res_prom)**2))

        # 2. Gaussiano
        t1 = time.time()
        res_gauss = cv.GaussianBlur(frame[220:450, 100:900], (n, n), 0)
        tiempos['Gaussiano'].append(time.time() - t1)
        mse_total['Gaussiano'].append(np.mean((frame[220:450, 100:900] - res_gauss)**2))

        # 3. Frecuencial
        t2 = time.time()
        radio = int(500 / n) # Ajuste empírico del radio
        res_freq = filtro_frecuencial_pasabajas(img_cortada, radio)
        tiempos['Frecuencial'].append(time.time() - t2)
        mse_total['Frecuencial'].append(np.mean((img_cortada - res_freq)**2))

        # resultadofinal
        final = frame.copy()
        # Aplicar blur solo en la máscara del ROI
        roi_color = frame[220:450, 100:900].copy()
        blur_color = cv.GaussianBlur(roi_color, (n, n), 0)
        
        # Mezclar usando la máscara
        roi_final = np.where(mascara_roi[:,:,None] == 255, blur_color, roi_color)
        final[220:450, 100:900] = roi_final
        
        cv.imshow("Anonimizacion de Codigo", final)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    # Resultados para la Tabla
    for tipo in ['Promedio', 'Gaussiano', 'Frecuencial']:
        m_mse = np.mean(mse_total[tipo])
        m_time = np.mean(tiempos[tipo])
        porcentaje = (m_mse / mse_max) * 100 if mse_max != 0 else 0
        print(f"{tipo} | MSE: {m_mse:.2f} | % Difuminación: {porcentaje:.2f}% | Tiempo: {m_time:.4f}s")

    video.release()
cv.destroyAllWindows()