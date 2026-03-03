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

#  obtener el frame de la mitad del video y calcular mse_max_ref
video = cv.VideoCapture(video_path)
total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
mid_frame = total_frames // 2
video.set(cv.CAP_PROP_POS_FRAMES, mid_frame)
ret, frame_ref = video.read()
if not ret:
    print("Error al leer el frame de referencia")
    exit()

# ROI del código en el frame de referencia
roi_ref = frame_ref[220:450, 100:900]
roi_ref_gris = cv.cvtColor(roi_ref, cv.COLOR_BGR2GRAY)

# Segmentar código en el frame de referencia
thresholds = threshold_multiotsu(roi_ref_gris, classes=4)
labels_ref = np.digitize(roi_ref_gris, bins=thresholds)
mascara_ref = np.zeros(roi_ref_gris.shape, dtype=np.uint8)
mascara_ref[(labels_ref == 1) | (labels_ref == 3)] = 255
codigo_pixels_ref = mascara_ref == 255

# Calcular máxima difuminación(47x47) en el frame de referencia
roi_ref_max_blur = cv.blur(roi_ref, (K_MAX, K_MAX))
orig_codigo_ref = roi_ref[codigo_pixels_ref]
max_codigo_ref = roi_ref_max_blur[codigo_pixels_ref]
mse_max_ref = np.mean((orig_codigo_ref.astype(np.float32) - max_codigo_ref.astype(np.float32)) ** 2)
print(f"Frame de referencia (índice {mid_frame}): MSE_max = {mse_max_ref:.2f}")

video.release()

for n in ventanas:
    print(f"\nAnalizando Ventana {n}x{n} para Código de Barras")
    video = cv.VideoCapture(video_path)
    
    mse_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    tiempos = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    porcentaje_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # ROI del código de barras en el frame actual
        roi = frame[220:450, 100:900]
        roi_gris = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # Segmentación multi-Otsu para este frame
        thresholds = threshold_multiotsu(roi_gris, classes=4)
        labels = np.digitize(roi_gris, bins=thresholds)
        mascara = np.zeros(roi_gris.shape, dtype=np.uint8)
        mascara[(labels == 1) | (labels == 3)] = 255
        codigo_pixels = mascara == 255
        cv.imshow("Mascara Codigos", mascara)
        
        orig_codigo = roi[codigo_pixels]
        
        # 1. Promedio
        t0 = time.time()
        res_prom = cv.blur(roi, (n, n))
        tiempos['Promedio'].append(time.time() - t0)
        prom_codigo = res_prom[codigo_pixels]
        mse_prom = np.mean((orig_codigo.astype(np.float32) - prom_codigo.astype(np.float32)) ** 2)
        mse_total['Promedio'].append(mse_prom)
        porcentaje_total['Promedio'].append((mse_prom / mse_max_ref) * 100 if mse_max_ref != 0 else 0)
        
        # 2. Gaussiano
        t1 = time.time()
        res_gauss = cv.GaussianBlur(roi, (n, n), 0)
        tiempos['Gaussiano'].append(time.time() - t1)
        gauss_codigo = res_gauss[codigo_pixels]
        mse_gauss = np.mean((orig_codigo.astype(np.float32) - gauss_codigo.astype(np.float32)) ** 2)
        mse_total['Gaussiano'].append(mse_gauss)
        porcentaje_total['Gaussiano'].append((mse_gauss / mse_max_ref) * 100 if mse_max_ref != 0 else 0)
        
        # 3. Frecuencial
        t2 = time.time()
        radio = int(500 / n)
        res_freq_gris = filtro_frecuencial_pasabajas(roi_gris, radio)
        res_freq = cv.cvtColor(res_freq_gris, cv.COLOR_GRAY2BGR)
        tiempos['Frecuencial'].append(time.time() - t2)
        freq_codigo = res_freq[codigo_pixels]
        mse_freq = np.mean((orig_codigo.astype(np.float32) - freq_codigo.astype(np.float32)) ** 2)
        mse_total['Frecuencial'].append(mse_freq)
        porcentaje_total['Frecuencial'].append((mse_freq / mse_max_ref) * 100 if mse_max_ref != 0 else 0)
        
        # Visualización: blur del código (usamos gaussiano para mostrar)
        final = frame.copy()
        roi_final = roi.copy()
        roi_final[codigo_pixels] = res_gauss[codigo_pixels]
        final[220:450, 100:900] = roi_final
        
        cv.imshow(f"Anonimizacion de Codigo {n}x{n}", final)
        cv.imshow("Imagen original", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv.destroyAllWindows()
    
    # Reporte de métricas para la Tabla 3
    print("\nTabla de resultados - Ventana {}x{} (Codigo de barras)".format(n, n))
    print("{:<12} | {:>10} | {:>12} | {:>10} | {}".format("Filtro", "MSE", "Porcentaje(%)", "Tiempo(s)", "Imagen"))
    print("-" * 70)
    for tipo in ['Promedio', 'Gaussiano', 'Frecuencial']:
        if len(mse_total[tipo]) > 0:
            m_mse = np.mean(mse_total[tipo])
            m_porc = np.mean(porcentaje_total[tipo])
            m_time = np.mean(tiempos[tipo])
        else:
            m_mse = m_porc = m_time = 0
        print("{:<12} | {:>10.2f} | {:>12.2f} | {:>10.4f} | {}".format(tipo, m_mse, m_porc, m_time, "mostrada"))