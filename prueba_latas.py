import cv2 as cv
import numpy as np
import time

def aplicar_filtro_frecuencial(img_gris, radio):
    # Transformada de Fourier
    dft = cv.dft(np.float32(img_gris), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img_gris.shape
    crow, ccol = rows // 2, cols // 2
    
    # Crear máscara pasa-bajas (Círculo blanco en centro negro)
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv.circle(mask, (ccol, crow), radio, (1, 1, 1), -1)
    
    # Aplicar máscara y transformar de vuelta
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    return cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Configuración inicial
video_path = 'videos/videos renombrar/2_Latas_orden.mp4'
ventanas = [9, 13, 21]
K_MAX = 47 

for n in ventanas:
    print(f"\nAnalizando Ventana {n}x{n} ")
    video = cv.VideoCapture(video_path)
    mse_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    tiempo_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    
    ret, frame_ref = video.read()
    if ret:
        frame_ref_gris = cv.cvtColor(frame_ref, cv.COLOR_BGR2GRAY)
        ref_blur = cv.blur(frame_ref_gris, (K_MAX, K_MAX))
        mse_max = np.mean((frame_ref_gris - ref_blur) ** 2)
    
    video.set(cv.CAP_PROP_POS_FRAMES, 0) # Reiniciar video

    while video.isOpened():
        ret, frame = video.read()
        if not ret: break
        
        # ROI quitando el fondo
        mascara_banda = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.rectangle(mascara_banda, (100, 200), (900, 500), 255, -1)
        mascara_fondo = cv.bitwise_not(mascara_banda)
        
        frame_gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        #Promedio filtro
        t0 = time.time()
        img_prom = cv.blur(frame, (n, n))
        tiempo_total['Promedio'].append(time.time() - t0)
        mse_p = np.mean((frame - img_prom)**2)
        mse_total['Promedio'].append(mse_p)

        #Gaussiano filtro
        t1 = time.time()
        img_gauss = cv.GaussianBlur(frame, (n, n), 0)
        tiempo_total['Gaussiano'].append(time.time() - t1)
        mse_g = np.mean((frame - img_gauss)**2)
        mse_total['Gaussiano'].append(mse_g)

        # Pasa bajas, filtro frecuencial
        radio = int(1000 / n) 
        t2 = time.time()
        img_freq_gris = aplicar_filtro_frecuencial(frame_gris, radio)
        tiempo_total['Frecuencial'].append(time.time() - t2)
        # Para comparar MSE con original, usamos frame_gris
        mse_f = np.mean((frame_gris - img_freq_gris)**2)
        mse_total['Frecuencial'].append(mse_f)

        # Construcción visual 
        res_final = frame.copy()
        # Aplicar el blur solo donde la máscara de fondo es 255 
        res_final = np.where(mascara_fondo[:,:,None] == 255, img_gauss, frame)
        
        cv.imshow(f"Fondo Nublado {n}x{n}", res_final)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    # Reporte de métricas para la Tabla 1 
    for tipo in ['Promedio', 'Gaussiano', 'Frecuencial']:
        m_mse = np.mean(mse_total[tipo])
        m_time = np.mean(tiempo_total[tipo])
        porcentaje = (m_mse / mse_max) * 100 
        print(f"{tipo} MSE: {m_mse:.2f} | % Difuminación: {porcentaje:.2f}% | Tiempo: {m_time:.4f}s")

    video.release()
cv.destroyAllWindows()