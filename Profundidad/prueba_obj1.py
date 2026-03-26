import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    archivo_entrada = 'juntas_ordenadas.bag'
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(archivo_entrada)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    # --- PARÁMETROS DE REFERENCIA Y ROIS ---
    DIST_REFERENCIA = 54.8  # cm
    
    # ROI Visualización (Donde se mantiene la etiqueta)
    roi_v = {'x': 230, 'y': 30, 'w': 200, 'h': 250}
    roi_c = {'x': 230, 'y': 210, 'w': 200, 'h': 80}
    
    umbral_altura_minima = 0.5  # cm
    
    # --- VARIABLES DE ESTADO PARA PERMANENCIA ---
    etiqueta_retenida = "BANDA VACIA"
    altura_retenida = 0.0
    color_display = (0, 0, 255) # Rojo por defecto (vacio)

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
            
            # Conversión a cm
            depth_cm = depth_image * depth_frame.get_units() * 100
            alturas = np.where((depth_cm > 0), DIST_REFERENCIA - depth_cm, 0)

            # 1. DETECCIÓN EN ROI_C (ZONA DE CÁLCULO)
            mask_c = np.zeros(depth_cm.shape, dtype=np.uint8)
            mask_c[roi_c['y']:roi_c['y']+roi_c['h'], roi_c['x']:roi_c['x']+roi_c['w']] = \
                (alturas[roi_c['y']:roi_c['y']+roi_c['h'], roi_c['x']:roi_c['x']+roi_c['w']] > umbral_altura_minima).astype(np.uint8) * 255
            
            contornos_c, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contornos_c:
                if cv2.contourArea(cnt) > 500:
                    # Si hay algo en el área de cálculo, actualizamos los datos retenidos
                    x, y, w, h = cv2.boundingRect(cnt)
                    dist_v = depth_frame.get_distance(x + w//2, y + h//2) * 100
                    altura_actual = DIST_REFERENCIA - dist_v if dist_v > 0 else 0
                    
                    altura_retenida = altura_actual
                    etiqueta_retenida = "Grande" if altura_actual > 12 else "Mediana" if altura_actual > 7 else "Chica"
                    color_display = (0, 255, 0) # Verde (detectado)

            # 2. VERIFICACIÓN DE SALIDA EN ROI_V (ÁREA DE VISUALIZACIÓN)
            # Creamos una máscara para ver si todavía hay "masa" de objeto en la zona visual
            mask_v = np.zeros(depth_cm.shape, dtype=np.uint8)
            mask_v[roi_v['y']:roi_v['y']+roi_v['h'], roi_v['x']:roi_v['x']+roi_v['w']] = \
                (alturas[roi_v['y']:roi_v['y']+roi_v['h'], roi_v['x']:roi_v['x']+roi_v['w']] > umbral_altura_minima).astype(np.uint8) * 255
            
            contornos_v, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hay_presencia_visual = False
            for cnt in contornos_v:
                if cv2.contourArea(cnt) > 500:
                    hay_presencia_visual = True
                    # Dibujamos el Bounding Box dinámico del objeto mientras esté en esta zona
                    vx, vy, vw, vh = cv2.boundingRect(cnt)
                    cv2.rectangle(color_image, (vx, vy), (vx + vw, vy + vh), color_display, 2)
                    cv2.circle(color_image, (vx + vw//2, vy + vh//2), 5, (0, 0, 255), -1)
                    break

            # 3. LÓGICA DE PERSISTENCIA
            # Si ya no hay presencia en la zona visual, reseteamos la etiqueta
            if not hay_presencia_visual:
                etiqueta_retenida = "BANDA VACIA"
                altura_retenida = 0.0
                color_display = (0, 0, 255)

            # Mostrar la etiqueta en la parte superior del ROI_V
            texto_pantalla = f"{etiqueta_retenida}" if altura_retenida == 0 else f"{etiqueta_retenida} H:{altura_retenida:.1f}cm"
            cv2.putText(color_image, texto_pantalla, (roi_v['x'], roi_v['y'] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_display, 2)

            # Dibujar Guías de los ROIs
            cv2.rectangle(color_image, (roi_v['x'], roi_v['y']), (roi_v['x'] + roi_v['w'], roi_v['y'] + roi_v['h']), (255, 255, 0), 1)
            cv2.rectangle(color_image, (roi_c['x'], roi_c['y']), (roi_c['x'] + roi_c['w'], roi_c['y'] + roi_c['h']), (0, 255, 255), 1)
            cv2.putText(color_image, "VISUALIZACION", (roi_v['x']+5, roi_v['y']+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(color_image, "CALCULO", (roi_c['x']+5, roi_c['y']+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow('Practica 1 - Clasificacion con Persistencia', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()