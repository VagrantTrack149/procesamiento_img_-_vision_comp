import cv2 as cv
import numpy as np
import time

def aplicar_filtro_frecuencial(img_gris, radio):
    h, w = img_gris.shape
    img_float = np.float32(img_gris)
    
    dft = cv.dft(img_float, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    crow, ccol = h // 2, w // 2
    
    mask = np.zeros((h, w), dtype=np.float32)
    cv.circle(mask, (ccol, crow), radio, 1, -1)
    
    dft_shift[:,:,0] *= mask
    dft_shift[:,:,1] *= mask
    
    f_ishift = np.fft.ifftshift(dft_shift)
    
    img_back = cv.idft(f_ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back
# Configuración

#video_path = 'videos/videos renombrar/2_faltante.mp4'
video_path = 'videos/videos renombrar/2_Latas_orden.mp4'
#video_path = 'videos/videos renombrar/2_latas_desorden.mp4'

ventanas = [9, 13, 21]
K_MAX = 47

# Obtener el frame de la mitad del video y calcular el MSE máximo como referencia
video_temp = cv.VideoCapture(video_path)
total_frames = int(video_temp.get(cv.CAP_PROP_FRAME_COUNT))
target_frame = total_frames // 2
video_temp.set(cv.CAP_PROP_POS_FRAMES, target_frame)
ret, frame_ref = video_temp.read()
if not ret:
    print("Error: no se pudo leer el frame de referencia")
    exit()
video_temp.release()

# Calcular máscara de fondo para el frame de referencia
mascara_banda_ref = np.zeros(frame_ref.shape[:2], dtype=np.uint8)
cv.rectangle(mascara_banda_ref, (100, 200), (900, 500), 255, -1)
mascara_fondo_ref = cv.bitwise_not(mascara_banda_ref)
fondo_pixels_ref = mascara_fondo_ref == 255

# Aplicar filtro de máxima difuminación (47x47) solo sobre el fondo
img_max_blur_ref = cv.blur(frame_ref, (K_MAX, K_MAX))
orig_fondo_ref = frame_ref[fondo_pixels_ref]
max_fondo_ref = img_max_blur_ref[fondo_pixels_ref]

mse_max_ref = np.mean((orig_fondo_ref.astype(np.float32) - max_fondo_ref.astype(np.float32)) ** 2)
print(f"MSE máximo de referencia (frame {target_frame}): {mse_max_ref:.2f}")

for n in ventanas:
    print(f"\nAnalizando Ventana {n}x{n}")
    video = cv.VideoCapture(video_path)
    # Diccionarios para acumular métricas
    mse_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    tiempo_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}
    porcentaje_total = {'Promedio': [], 'Gaussiano': [], 'Frecuencial': []}

    frame_count = 0
    imagen_bordes = None  # guardará el frame con fondo difuminado para la parte 2

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Máscara del fondo
        mascara_banda = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.rectangle(mascara_banda, (100, 200), (900, 500), 255, -1)
        mascara_fondo = cv.bitwise_not(mascara_banda)
        fondo_pixels = mascara_fondo == 255

        orig_fondo = frame[fondo_pixels]

        # Filtro promedio
        t0 = time.time()
        img_prom = cv.blur(frame, (n, n))
        tiempo_total['Promedio'].append(time.time() - t0)
        prom_fondo = img_prom[fondo_pixels]
        mse_prom = np.mean((orig_fondo.astype(np.float32) - prom_fondo.astype(np.float32)) ** 2)
        mse_total['Promedio'].append(mse_prom)
        porcentaje_total['Promedio'].append((mse_prom / mse_max_ref) * 100 if mse_max_ref != 0 else 0)

        # Filtro gaussiano
        t1 = time.time()
        img_gauss = cv.GaussianBlur(frame, (n, n), 0)
        tiempo_total['Gaussiano'].append(time.time() - t1)
        gauss_fondo = img_gauss[fondo_pixels]
        mse_gauss = np.mean((orig_fondo.astype(np.float32) - gauss_fondo.astype(np.float32)) ** 2)
        mse_total['Gaussiano'].append(mse_gauss)
        porcentaje_total['Gaussiano'].append((mse_gauss / mse_max_ref) * 100 if mse_max_ref != 0 else 0)

        # Filtro frecuencial (pasa bajas)
        radio = int(1000 / n)
        t2 = time.time()
        frame_gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img_freq_gris = aplicar_filtro_frecuencial(frame_gris, radio)
        img_freq = cv.cvtColor(img_freq_gris, cv.COLOR_GRAY2BGR)  # convertir a color para mantener dimensiones
        tiempo_total['Frecuencial'].append(time.time() - t2)
        freq_fondo = img_freq[fondo_pixels]
        mse_freq = np.mean((orig_fondo.astype(np.float32) - freq_fondo.astype(np.float32)) ** 2)
        mse_total['Frecuencial'].append(mse_freq)
        porcentaje_total['Frecuencial'].append((mse_freq / mse_max_ref) * 100 if mse_max_ref != 0 else 0)

        #fondo difuminado gauss
        res_final = frame.copy()
        res_final[fondo_pixels] = img_gauss[fondo_pixels]

        cv.imshow(f"Fondo Nublado {n}x{n}", res_final)

        # Detección de bordes
        #bordes = cv.Canny(res_final, 100, 200)
        #cv.imshow(f"Bordes {n}x{n}", bordes)

        # Guardar el frame de la mitad para la parte 2
        if frame_count == target_frame:
            imagen_bordes = res_final.copy()

        frame_count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()

    # Reporte de métricas para la Tabla 1
    print("\nTabla de resultados - Ventana {}x{}".format(n, n))
    print("{:<12} | {:>10} | {:>12} | {:>10} | {}".format("Filtro", "MSE", "Porcentaje(%)", "Tiempo(s)", "Imagen"))
    for tipo in ['Promedio', 'Gaussiano', 'Frecuencial']:
        m_mse = np.mean(mse_total[tipo])
        m_porc = np.mean(porcentaje_total[tipo])
        m_time = np.mean(tiempo_total[tipo])
        print("{:<12} | {:>10.2f} | {:>12.2f} | {:>10.4f} | {}".format(tipo, m_mse, m_porc, m_time, "mostrada en ventana"))

    # Parte 2: Detección de bordes en la imagen de la mitad del video
    if imagen_bordes is not None:
        print("Parte 2 Comparación de filtros pasa altas")
        img_gris = cv.cvtColor(imagen_bordes, cv.COLOR_BGR2GRAY)

        filtros = {
            'Canny': lambda im: cv.Canny(im, 100, 200),
            'Sobel': lambda im: cv.convertScaleAbs(cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3) + cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)),
            'Laplaciano': lambda im: cv.convertScaleAbs(cv.Laplacian(im, cv.CV_64F)),
            'Scharr': lambda im: cv.convertScaleAbs(cv.Scharr(im, cv.CV_64F, 1, 0) + cv.Scharr(im, cv.CV_64F, 0, 1))
        }

        print("\nTabla de resultados - Bordes (imagen única)")
        print("{:<12} | {:>10} | {:>12} | {:>10} | {}".format("Filtro", "MSE", "Porcentaje(%)", "Tiempo(s)", "Imagen"))
        for nombre, func in filtros.items():
            t0 = time.time()
            bordes_img = func(img_gris)
            t = time.time() - t0

            mse = np.mean(bordes_img.astype(np.float32) ** 2)
            porcentaje = (mse / (255*255)) * 100

            print("{:<12} | {:>10.2f} | {:>12.2f} | {:>10.4f} | {}".format(nombre, mse, porcentaje, t, "mostrada"))
            cv.imshow(f"Bordes - {nombre}", bordes_img)

        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("No se pudo capturar la imagen para la parte 2")

cv.destroyAllWindows()