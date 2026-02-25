import cv2 as cv
import numpy as np
venta_num=9
mse_lista= []
psnr_lista= []

video= cv.VideoCapture('videos/videos renombrar/2_Latas_orden.mp4')
frame_speed=33
while video.isOpened():
    ret, frame= video.read()
    if ret:
        cv.imshow("Video", frame)
        #cortando la imagen en la parte de las latas
        img_cortada= frame[200:500, 100:900]
        cv.imshow("Imagen Cortada", img_cortada)
        #mascara para quitar la parte de las latas y quedarnos con el fondo
        mascara= np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.rectangle(mascara, (100, 200), (900, 500), (255, 255,255), -1)
        img_mascara= cv.bitwise_and(frame, frame, mask=mascara)
        #img_mascara= cv.bitwise_not(frame, frame, mask=mascara)
        cv.imshow("Imagen Mascara", img_mascara)

        #aplicando la máscara invertida para remover las latas
        mascara_inv = cv.bitwise_not(mascara)
        fondo = cv.bitwise_and(frame, frame, mask=mascara_inv)
        cv.imshow("Fondo", fondo)
        #filtro gausiano
        img_gaussiana= cv.GaussianBlur(fondo, (venta_num,venta_num), 1) #ventana de 9 x 9 
        cv.imshow("Imagen Gaussiana", img_gaussiana)
        #filtro promedio gaussiano
        img_promedio_gaus= cv.blur(img_gaussiana, (venta_num,venta_num)) #ventana de 9 x 9
        cv.imshow("Imagen Promedio Gaussiana", img_promedio_gaus)

        #fusionar imagenes 
        resultado_final= img_promedio_gaus.copy()
        resultado_final[200:500, 100:900] = frame[200:500, 100:900]
        cv.imshow("Resultado Final", resultado_final)
        #calculo MSE Y PSNR
        mse= np.mean((frame - img_promedio_gaus)**2)
        
        mse_lista.append(mse)   
        #promediar mse para obtener un valor más estable

        #print("MSE:", mse)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255**2 / mse)
        psnr_lista.append(psnr)
        #promediar psnr para obtener un valor más estable

        #print("PSNR:", psnr)
    if not ret:
        break
    if cv.waitKey(frame_speed) & 0xFF == ord('q'):    
        break
print("MSE Promedio:", np.mean(mse_lista))
print("PSNR Promedio:", np.mean(psnr_lista))
video.release()
cv.destroyAllWindows()