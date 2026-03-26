import pyrealsense2 as rs
import numpy as np
import cv2 as cv

def main():
    #archivo_entrada = 'juntas_ordenadas.bag'
    archivo_entrada = 'juntas_abolladas.bag'

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(archivo_entrada)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    DIST_REFERENCIA = 54.8 
    roi_x, roi_y, roi_w, roi_h = 230, 30, 215, 250
    umbral_altura_minima = 0.5 
    
    UMBRAL_ABOLLADURA = 0.4 

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_cm = depth_image * depth_frame.get_units() * 100

            
            alturas = np.where((depth_cm > 0), DIST_REFERENCIA - depth_cm, 0)
            mask = np.zeros(depth_cm.shape, dtype=np.uint8)
            mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = \
                (alturas[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] > umbral_altura_minima).astype(np.uint8) * 255

            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv.contourArea(cnt) > 800:
                    x, y, w, h = cv.boundingRect(cnt)
                    
                    superficie = alturas[y:y+h, x:x+w]
                    pixeles_validos = superficie[superficie > umbral_altura_minima]
                    
                    if len(pixeles_validos) > 0:
                        uniformidad = np.std(pixeles_validos) # Desviación estándar
                        altura_media = np.mean(pixeles_validos)
                        
                        # Determinar estado
                        if uniformidad > UMBRAL_ABOLLADURA:
                            estado = "ABOLLADA"
                            color_status = (0, 0, 255) # Rojo
                        else:
                            estado = "NORMAL"
                            color_status = (0, 255, 0) # Verde

                        
                        cv.rectangle(color_image, (x, y), (x + w, y + h), color_status, 2)
                        cv.putText(color_image, f"{estado}", (x, y - 25), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
                        cv.putText(color_image, f"Var: {uniformidad:.2f}cm", (x, y - 5), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        
                        cv.circle(color_image, (x + w//2, y + h//2), 4, (255, 0, 0), -1)

            cv.rectangle(color_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 1)
            cv.imshow('Analisis de Defectos (Profundidad) - Practica 1', color_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()