import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Configuración de la cámara
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciar el pipeline
pipeline.start(config)

print("Cámara lista. Presiona 's' para guardar y 'q' para salir.")

if not os.path.exists('capturas'):
    os.makedirs('capturas')

count = 0

try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow('RealSense Video', color_image)
        
        key = cv2.waitKey(1)
        
        # Si presionas 's', se guarda la imagen
        if key & 0xFF == ord('s'):
            filename = f"capturas/foto_{count}.jpg"
            cv2.imwrite(filename, color_image)
            print(f"Imagen guardada como: {filename}")
            count += 1
            
        # Si presionas 'q', se cierra el programa
        elif key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()