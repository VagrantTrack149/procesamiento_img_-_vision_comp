import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

def guardar_datos_txt(datos):
    """Guarda una línea con los detalles técnicos en un archivo de texto"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = (f"{timestamp} | Clase: {datos['clase']} | Estado: {datos['estado']} | "
             f"H_Max: {datos['h_max']:.2f}cm | Borde: {datos['h_borde']:.2f}cm | "
             f"Centro: {datos['h_centro']:.2f}cm | Diff: {datos['delta']:.2f}cm | "
             f"Err: {datos['error']:.2f}cm\n")
    
    with open("abolladas.txt", "a") as f:
        f.write(linea)

def main():
    archivo_entrada = 'juntas_abolladas.bag' 
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(archivo_entrada)
    
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    # --- PARÁMETROS FÍSICOS (CM) ---
    DIST_REFERENCIA = 54.8 
    LATAS = {
        "GRANDE":  {"alt": 150.0, "diam": 5.4, "anillo": 4.6, "esperado": 1.0},
        "MEDIANA": {"alt": 105.0, "diam": 5.4, "anillo": 4.6, "esperado": 1.0},
        "CHICA":   {"alt": 2.0,  "diam": 8.5, "anillo": 7.6, "esperado": 0.5}
    }

    roi_c = {'x': 230, 'y': 210, 'w': 200, 'h': 80}
    roi_v = {'x': 230, 'y': 30,  'w': 200, 'h': 250}
    centro_y_target = roi_c['y'] + (roi_c['h'] // 2)
    
    lata_en_proceso = False 
    memoria = {
        "clase": "---", "estado": "ESPERANDO", 
        "h_max": 0.0, "h_borde": 0.0, "h_centro": 0.0, 
        "delta": 0.0, "error": 0.0, "color": (200, 200, 200)
    }

    def limpiar_outliers(datos):
        if datos.size < 4: return np.mean(datos) if datos.size > 0 else 0
        q1, q3 = np.percentile(datos, [25, 75])
        iqr = q3 - q1
        filtro = datos[(datos >= q1 - 1.5 * iqr) & (datos <= q3 + 1.5 * iqr)]
        return np.mean(filtro) if filtro.size > 0 else np.mean(datos)

    try:
        # Crear encabezado si el archivo es nuevo
        with open("registro_calidad.txt", "a") as f:
            f.write(f"\n--- INICIO DE SESIÓN: {datetime.now()} ---\n")

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f: continue

            color_img = np.asanyarray(color_f.get_data())
            depth_img = np.asanyarray(depth_f.get_data())
            depth_img = cv2.medianBlur(depth_img, 3)
            
            depth_cm = depth_img * depth_f.get_units() * 100
            alturas = np.where((depth_cm > 0) & (depth_cm < 100), DIST_REFERENCIA - depth_cm, 0)

            mask_c = np.zeros(alturas.shape, dtype=np.uint8)
            mask_c[roi_c['y']:roi_c['y']+roi_c['h'], roi_c['x']:roi_c['x']+roi_c['w']] = \
                (alturas[roi_c['y']:roi_c['y']+roi_c['h'], roi_c['x']:roi_c['x']+roi_c['w']] > 0.5).astype(np.uint8) * 255
            
            contornos, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objeto_detectado = False
            for cnt in contornos:
                if cv2.contourArea(cnt) > 1000:
                    objeto_detectado = True
                    M = cv2.moments(cnt)
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                    if not lata_en_proceso and (centro_y_target - 4 <= cy <= centro_y_target + 4):
                        lata_en_proceso = True 
                        
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        crop = alturas[by:by+bh, bx:bx+bw]
                        
                        h_max_robusta = np.mean(np.sort(crop.flatten())[-10:])
                        if h_max_robusta > 13.0: clase = "GRANDE"
                        elif h_max_robusta > 7.0: clase = "MEDIANA"
                        else: clase = "CHICA"
                        
                        specs = LATAS[clase]
                        lx, ly = cx - bx, cy - by

                        centro_vls = crop[max(0, ly-3):ly+3, max(0, lx-3):lx+3].flatten()
                        h_centro = limpiar_outliers(centro_vls)

                        mask_anillo = np.zeros(crop.shape, dtype=np.uint8)
                        r_px = (bw / 2) * (specs["anillo"] / specs["diam"])
                        cv2.circle(mask_anillo, (lx, ly), int(r_px), 255, 2)
                        h_borde = limpiar_outliers(crop[mask_anillo == 255])

                        diff = h_borde - h_centro
                        error_abs = abs(diff - specs["esperado"])

                        memoria.update({
                            "clase": clase, "h_max": h_max_robusta, "h_borde": h_borde,
                            "h_centro": h_centro, "delta": diff, "error": error_abs,
                            "estado": "ABOLLADA" if error_abs > 0.35 else "NORMAL",
                            "color": (0, 0, 255) if error_abs > 0.35 else (0, 255, 0)
                        })
                        
                        # --- GUARDAR EN TXT ---
                        guardar_datos_txt(memoria)

            if not objeto_detectado:
                lata_en_proceso = False
                memoria["estado"] = "ESPERANDO"

            # Visualización (Mismo código anterior)
            y_offset = roi_v['y']
            cv2.putText(color_img, f"{memoria['clase']} | {memoria['estado']}", (roi_v['x'], y_offset - 10), 1, 1.3, memoria["color"], 2)
            datos_pantalla = [
                f"H. Max: {memoria['h_max']:.2f} cm", f"H. Borde: {memoria['h_borde']:.2f} cm",
                f"H. Centro: {memoria['h_centro']:.2f} cm", f"Diff (B-C): {memoria['delta']:.2f} cm",
                f"Error: {memoria['error']:.2f} cm"
            ]
            for i, dato in enumerate(datos_pantalla):
                cv2.putText(color_img, dato, (roi_v['x'] + 5, y_offset + 25 + (i*20)), 1, 0.8, (255, 255, 255), 1)

            cv2.imshow('Control de Calidad - TXT Activo', color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()