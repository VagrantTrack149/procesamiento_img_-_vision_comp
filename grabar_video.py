import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# Configuración de la cámara
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciar el pipeline
pipeline.start(config)

print("Cámara lista.")
print("Presiona 'r' para iniciar/detener grabación")
print("Presiona 'q' para salir")

if not os.path.exists('videos'):
    os.makedirs('videos')

# Configurar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False
video_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Si está grabando, escribir el frame
        if recording and out is not None:
            out.write(color_image)

        # Mostrar estado de grabación en la imagen
        if recording:
            cv2.putText(color_image, "GRABANDO...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('RealSense Video', color_image)
        
        key = cv2.waitKey(1)
        
        # Si presionas 'r', iniciar/detener grabación
        if key & 0xFF == ord('r'):
            if not recording:
                # Iniciar grabación
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"videos/video_{timestamp}.mp4"
                out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
                recording = True
                print(f"Grabación iniciada: {filename}")
            else:
                # Detener grabación
                if out is not None:
                    out.release()
                    out = None
                recording = False
                print("Grabación detenida")
            
        # Si presionas 'q', salir
        elif key & 0xFF == ord('q'):
            break

finally:
    if out is not None:
        out.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Programa finalizado")
