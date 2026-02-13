import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt

#captura de camara
cap= cv.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if not ret:
        print("No se pudo capturar el video")
        break
    cv.imshow("Camara", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()