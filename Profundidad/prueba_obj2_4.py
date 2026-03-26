import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime

def clasificar_lata_avanzado(clase, h_aros):
    if len(h_aros) < 4 or any(h == 0 for h in h_aros):
        return "Error", (255, 255, 255), 0.0

    a0, a1, a2, a3 = h_aros
    delta = a3 - a0  # Borde menos Centro
    
    # 1. Delta (detectar domo xd)
    umbrales_delta = {
        "GRANDE":  -0.18,
        "MEDIANA": -0.20,
        "CHICA":   -0.5   # ajustar - no funciona pero ya me cansé xdxd
    }
    fail_delta = delta < umbrales_delta.get(clase, -0.20)

    # 2 Pendiente (Busca cambios extremos)
    saltos = [abs(a1 - a0), abs(a2 - a1), abs(a3 - a2)]
    max_salto = max(saltos)
    fail_pendiente = max_salto > 0.35   # keep as is for now

    # 3 Varianza para medir si es irregular
    std_dev = np.std(h_aros)
    umbral_std = {
        "GRANDE":  0.073,
        "MEDIANA": 0.110,
        "CHICA":   0.35    # prueba
    }
    fail_varianza = std_dev > umbral_std.get(clase, 0.25)

    #  final
    if fail_delta:
        return "ABOLLADA (DOMO)", (0, 0, 255), std_dev
    elif fail_pendiente:
        return "ABOLLADA (SALTO)", (0, 0, 255), std_dev
    elif fail_varianza:
        return "ABOLLADA (RUGOSA)", (0, 0, 255), std_dev
    else:
        return "NORMAL", (0, 255, 0), std_dev

def guardar_registro(m):
    #luego quitar guardar datos para prueba
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = (
        f"{timestamp} | {m['clase']:7} | {m['estado']:17} | "
        f"D:{m['delta']:5.2f} | STD:{m['std_dev']:5.2f} | V:{m['vel']:6.2f} | "
        f"A0:{m['h_aros'][0]:5.2f} | A1:{m['h_aros'][1]:5.2f} | "
        f"A2:{m['h_aros'][2]:5.2f} | A3:{m['h_aros'][3]:5.2f}\n"
    )
    with open("dataset_produccion.txt", "a") as f:
        #f.write(linea)
        print("hola")

def limpiar_outliers(datos):
    if datos.size < 4: return np.mean(datos) if datos.size > 0 else 0
    q1, q3 = np.percentile(datos, [25, 75])
    iqr = q3 - q1
    filtro = datos[(datos >= q1 - 1.5 * iqr) & (datos <= q3 + 1.5 * iqr)]
    return np.mean(filtro) if filtro.size > 0 else np.mean(datos)

def obtener_datos_aro(crop_alt, centro, radio_px, grosor=2):
    mask = np.zeros(crop_alt.shape, dtype=np.uint8)
    cv2.circle(mask, centro, int(radio_px), 255, grosor)
    puntos = crop_alt[mask == 255]
    return limpiar_outliers(puntos[puntos > 0])

def main():
    archivo_entrada = 'juntas_abolladas.bag' ################################################################################
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(archivo_entrada)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # PARAMETROS 
    DIST_REFERENCIA = 54.8 
    roi_v = {'x': 245, 'y': 30, 'w': 200, 'h': 250}
    roi_c = {'x': 245, 'y': 220, 'w': 200, 'h': 60}
    LINEA_DISPARO = roi_c['y'] + (roi_c['h'] // 2)

    memoria = {
        "clase": "ESPERANDO", "estado": "---", "vel": 0.0, 
        "pos_y": 0, "t": time.time(), "h_aros": [0,0,0,0], "delta": 0,
        "evaluado": False, "h_max": 0, "std_dev": 0, "color": (255,255,255)
    }

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f: continue

            color_img = np.asanyarray(color_f.get_data())
            depth_cm = np.asanyarray(depth_f.get_data()) * depth_f.get_units() * 100
            alturas = np.where((depth_cm > 0), DIST_REFERENCIA - depth_cm, 0)

            # Busqueda de objetos en ROI_V
            mask_v = (alturas[roi_v['y']:roi_v['y']+roi_v['h'], roi_v['x']:roi_v['x']+roi_v['w']] > 0.5).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hay_objeto = False
            for cnt in cnts:
                if cv2.contourArea(cnt) > 800:
                    hay_objeto = True
                    M = cv2.moments(cnt)
                    cx, cy = int(M["m10"]/M["m00"]) + roi_v['x'], int(M["m01"]/M["m00"]) + roi_v['y']
                    
                    dt = time.time() - memoria["t"]
                    dy = cy - memoria["pos_y"]
                    vel_actual = abs(dy) / dt if dt > 0 else 0
                    
                    # Línea media de calculo
                    if abs(cy - LINEA_DISPARO) < 5 and not memoria["evaluado"]:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        bx, by = bx + roi_v['x'], by + roi_v['y']
                        crop_alt = alturas[by:by+bh, bx:bx+bw]
                        
                        h_max = np.percentile(crop_alt, 98)
                        clase = "GRANDE" if h_max > 13 else "MEDIANA" if h_max > 7 else "CHICA"
                        
                        # Cálculo de los 4 aros
                        centro_lata = (bw // 2, bh // 2)
                        radios = [int(bw*0.15), int(bw*0.25), int(bw*0.35), int(bw*0.45)]
                        h_aros = [obtener_datos_aro(crop_alt, centro_lata, r) for r in radios]
                        
                        
                        estado, color, std_dev = clasificar_lata_avanzado(clase, h_aros)
                        
                        memoria.update({
                            "clase": clase, "estado": estado, "h_max": h_max,
                            "h_aros": h_aros, "delta": h_aros[3] - h_aros[0],
                            "color": color, "std_dev": std_dev,
                            "evaluado": True, "vel": vel_actual, "radios_px": radios
                        })
                        
                        guardar_registro(memoria)
                        print(f"[OK] Clasificado como: {estado}")

                    # persiste
                    vx, vy, vw, vh = cv2.boundingRect(cnt)
                    vx, vy = vx + roi_v['x'], vy + roi_v['y']
                    cv2.rectangle(color_img, (vx, vy), (vx+vw, vy+vh), memoria["color"], 2)
                    
                    #mostrar info
                    tx = vx + vw + 10
                    cv2.putText(color_img, f"{memoria['clase']}", (tx, vy), 1, 1.2, (255,255,255), 2)
                    cv2.putText(color_img, f"{memoria['estado']}", (tx, vy + 25), 1, 1.0, memoria["color"], 2)
                    cv2.putText(color_img, f"D: {memoria['delta']:.2f} STD: {memoria['std_dev']:.2f}", (tx, vy + 50), 1, 0.8, (0,255,255), 1)

                    # Dibujar aros si ya fue evaluada
                    if memoria["evaluado"] and "radios_px" in memoria:
                        for r in memoria["radios_px"]:
                            cv2.circle(color_img, (cx, cy), r, (255, 255, 0), 1)

                    memoria.update({"pos_y": cy, "t": time.time()})

            if not hay_objeto:
                memoria["evaluado"] = False
                memoria["clase"] = "ESPERANDO"
                memoria["color"] = (255,255,255)

            # areas
            cv2.line(color_img, (roi_c['x'], LINEA_DISPARO), (roi_c['x'] + roi_c['w'], LINEA_DISPARO), (0, 0, 255), 2)
            cv2.rectangle(color_img, (roi_v['x'], roi_v['y']), (roi_v['x']+roi_v['w'], roi_v['y']+roi_v['h']), (100,100,100), 1)
            
            cv2.imshow('Sistema de Clasificacion Avanzada', color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()