import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu

video= cv.VideoCapture('videos/videos renombrar/2_codigo_amarillol.mp4')
frame_speed=33
ksize=(9,9)
while video.isOpened():
    ret, frame= video.read()
    if ret:
        cv.imshow("Video", frame)
        frame_gris=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        #cv.imshow("Video Gris", frame_gris)
        #detectando cuadro blanco en el espacio frame_gris[200:500, 100:900]
        img_cortada= frame_gris[220:450, 100:900]
        #cv.imshow("Imagen Cortada", img_cortada)
        #umbralización para detectar el código de barras
        img_umbralizada= threshold_multiotsu(img_cortada, classes=4)
        labels= np.digitize(img_cortada, bins=img_umbralizada)
        cv.imshow("Imagen Umbralizada", labels.astype(np.uint8)*85)
        #unicamente capa de codigos de barras de multiotsu
        mascara_codigos= np.zeros(frame.shape[:2], dtype=np.uint8)
        
        #mascara_codigos[220:450, 100:900][ (labels == 1)] = 255
        mascara_codigos[220:450, 100:900][ (labels == 3)] = 255
        
        cv.imshow("Mascara Codigos", mascara_codigos)
        mascara_temp_amarillo= np.zeros(frame.shape[:2], dtype=np.uint8)
        mascara_temp_amarillo[220:450, 100:900][ (labels == 2)] = 255
        #removiendo areas pequeñas de la mascara de amarillo
        mascara_temp_amarillo= cv.erode(mascara_temp_amarillo, np.ones((3,3), np.uint8), iterations=3)
        mascara_temp_amarillo = cv.dilate(mascara_temp_amarillo, np.ones((3,3), np.uint8), iterations=3)
        cv.imshow("Mascara Amarillo", mascara_temp_amarillo)
        #mascara_final= cv.bitwise_xor(mascara_codigos, mascara_temp_amarillo)
        #cv.imshow("Mascara Final", mascara_final)
        #blur en codigo de barras
        img_blur = cv.blur(mascara_codigos, ksize, cv.BORDER_DEFAULT) 
        kernel= np.ones(ksize, np.float32)/(ksize[0]*ksize[1])
        img_blur = cv.filter2D(img_blur, -1, kernel)
        img_blur = cv.GaussianBlur(img_blur, ksize, 0)
        #cv.imshow("Imagen Blur", img_blur)
        frame_blur = cv.blur(frame, ksize, cv.BORDER_DEFAULT)
        frame_blur = cv.filter2D(frame_blur, -1, kernel)
        
        final= frame.copy()
        final[220:450, 100:900]=cv.bitwise_and(frame[220:450, 100:900], frame_blur[220:450, 100:900], mask=cv.bitwise_or(mascara_codigos[220:450, 100:900], mascara_temp_amarillo[220:450, 100:900]))
        cv.imshow("Final", final)
        #resultado_final[220:450,100:900][ (labels == 1)] = frame_blur[220:450, 100:900][ (labels == 1)]
        #resultado_final[220:450, 100:900] = img_blur[220:450, 100:900][ (labels == 1)] 
        #resultado_final[220:450, 100:900][ (labels == 1)] = frame_blur[220:450, 100:900][ (labels == 1)]
        #resultado_final[220:450,100:900] = cv.bitwise_or(resultado_final[220:450,100:900], frame[220:450,100:900], mask=(mascara_temp_amarillo[220:450, 100:900]))
        #final=cv.GaussianBlur(final, ksize, 0)
        #cv.imshow("Final", final)
        #r_final= cv.bitwise_or(final, frame, mask=mascara_codigos)
        #cv.imshow("Resultado Final", r_final)
        #r_final= cv.GaussianBlur(r_final, ksize, 0)
        #cv.imshow("Resultado Final 1", r_final)
        #convertir r_final en una mascara binaria para detectar el código de barras
        #r_final_gris = cv.cvtColor(r_final, cv.COLOR_BGR2GRAY)
        
        #_, mask_bin = cv.threshold(r_final_gris, 1, 255, cv.THRESH_BINARY)
        #resultado_final = frame.copy()
        #cv.copyTo(r_final, mask_bin, resultado_final)
        #cv.imshow("Resultado Final 2", resultado_final)
        
        


    if not ret:
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()