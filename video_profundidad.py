import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# Configuración de la cámara
pipeline = rs.pipeline()
config = rs.config()

# Habilitar Color y Profundidad
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Res menor para ahorrar espacio

# Herramienta para alinear profundidad con color
align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)

if not os.path.exists('data'):
    os.makedirs('data')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False
depth_data_list = [] # Lista temporal para guardar matrices de proximidad

print("Cámara lista. 'r' para Grabar, 'q' para Salir.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # Alinear frames (crítico para que coincidan color y profundidad)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convertir a numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if recording:
            # Escribir video
            if out is not None:
                out.write(color_image)
            
            # Guardar datos de proximidad (en milímetros)
            depth_data_list.append(depth_image)
            
            cv2.putText(color_image, "GRABANDO DATA...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Visualización opcional de la profundidad (mapa de calor)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        cv2.imshow('RealSense Color', color_image)
        # cv2.imshow('RealSense Depth', depth_colormap) # Descomenta para ver la profundidad

        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('r'):
            if not recording:
                # Iniciar grabación
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = f"data/video_{timestamp}.mp4"
                
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
                depth_data_list = [] # Limpiar lista
                recording = True
                print(f"Grabando en: {video_path}")
            else:
                # Detener y Guardar
                recording = False
                if out is not None:
                    out.release()
                
                # Guardar datos de proximidad de forma COMPACTA (.npz)
                depth_path = f"data/depth_{timestamp}.npz"
                np.savez_compressed(depth_path, data=np.array(depth_data_list))
                
                print(f"Video guardado. Datos de proximidad guardados en: {depth_path}")
                print(f"Frames capturados: {len(depth_data_list)}")
            
        elif key & 0xFF == ord('q'):
            break

finally:
    if out is not None:
        out.release()
    pipeline.stop()
    cv2.destroyAllWindows()