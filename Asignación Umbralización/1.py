#extraer el primer fotograma de un video y mostrarlo
import cv2 as cv
import numpy as np
from skimage import filters

video_path = '1.mp4'  
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video")
    exit()
    
ret, frame = cap.read()

if not ret:
    print("Error al leer el fotograma")
    exit()

cv.imshow('Primer Fotograma Original', frame)
cv.waitKey(0)

cap.release()
cv.destroyAllWindows()