import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime

def clasificar_lata_avanzado(clase, h_aros):
    if len(h_aros) < 4 or any(h == 0 for h in h_aros):
        return "Error", (255, 255, 255), 0.0

    a0, a1, a2, a3 = h_aros
    delta = a3 - a0

    umbrales_delta = {
        "GRANDE":  -0.18,
        "MEDIANA": -0.20,
        "CHICA":   -0.5
    }
    fail_delta = delta < umbrales_delta.get(clase, -0.20)

    saltos = [abs(a1 - a0), abs(a2 - a1), abs(a3 - a2)]
    max_salto = max(saltos)
    fail_pendiente = max_salto > 0.35

    std_dev = np.std(h_aros)
    umbral_std = {
        "GRANDE":  0.073,
        "MEDIANA": 0.110,
        "CHICA":   0.35
    }
    fail_varianza = std_dev > umbral_std.get(clase, 0.25)

    if fail_delta:
        return "ABOLLADA (DOMO)", (0, 0, 255), std_dev
    elif fail_pendiente:
        return "ABOLLADA (SALTO)", (0, 0, 255), std_dev
    elif fail_varianza:
        return "ABOLLADA (RUGOSA)", (0, 0, 255), std_dev
    else:
        return "NORMAL", (0, 255, 0), std_dev


def guardar_registro(m):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = (
        f"{timestamp} | {m['clase']:7} | {m['estado']:17} | "
        f"D:{m['delta']:5.2f} | STD:{m['std_dev']:5.2f} | V:{m['vel']:6.2f} | "
        f"A0:{m['h_aros'][0]:5.2f} | A1:{m['h_aros'][1]:5.2f} | "
        f"A2:{m['h_aros'][2]:5.2f} | A3:{m['h_aros'][3]:5.2f}\n"
    )
    with open("dataset_produccion.txt", "a") as f:
        # f.write(linea)
        print("hola")


def limpiar_outliers(datos):
    if datos.size < 4:
        return np.mean(datos) if datos.size > 0 else 0
    q1, q3 = np.percentile(datos, [25, 75])
    iqr = q3 - q1
    filtro = datos[(datos >= q1 - 1.5 * iqr) & (datos <= q3 + 1.5 * iqr)]
    return np.mean(filtro) if filtro.size > 0 else np.mean(datos)


def obtener_datos_aro(crop_alt, centro, radio_px, grosor=2):
    mask = np.zeros(crop_alt.shape, dtype=np.uint8)
    cv2.circle(mask, centro, int(radio_px), 255, grosor)
    puntos = crop_alt[mask == 255]
    return limpiar_outliers(puntos[puntos > 0])



def dibujar_panel(img, vel_px_s, vel_m_s, memoria):
    
    panel_x = img.shape[1] - 640
    panel_y = 10
    panel_w = 150
    panel_h = 220

    overlay = img.copy()


    cv2.putText(img, "VELOCIDAD", (panel_x + 8, panel_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
    cv2.putText(img, f"{vel_px_s:7.1f} px/s", (panel_x + 8, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"{vel_m_s:7.3f}  m/s", (panel_x + 8, panel_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(img, (panel_x + 5, panel_y + 88), (panel_x + panel_w - 5, panel_y + 88),
             (60, 60, 60), 1)

    cv2.putText(img, "CLASIFICACION", (panel_x + 8, panel_y + 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
    cv2.putText(img, f"Tipo:   {memoria['clase']}", (panel_x + 8, panel_y + 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    color_estado = memoria["color"]
    cv2.putText(img, f"Estado: {memoria['estado']}", (panel_x + 8, panel_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_estado, 1)

    cv2.line(img, (panel_x + 5, panel_y + 150), (panel_x + panel_w - 5, panel_y + 150),
             (60, 60, 60), 1)

    cv2.putText(img, "METRICAS", (panel_x + 8, panel_y + 166),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    cv2.putText(img, f"Delta:  {memoria['delta']:5.3f}", (panel_x + 8, panel_y + 183),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"STD:    {memoria['std_dev']:5.3f}", (panel_x + 8, panel_y + 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"H.max:  {memoria['h_max']:5.2f} cm", (panel_x + 8, panel_y + 217),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    archivo_entrada = 'espaciadas_abolladas.bag'  ################################################################
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(archivo_entrada)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    #parametros
    DIST_REFERENCIA = 54.8   # cm
    roi_v = {'x': 245, 'y': 30,  'w': 200, 'h': 250}
    roi_c = {'x': 245, 'y': 220, 'w': 200, 'h': 60}
    LINEA_DISPARO = roi_c['y'] + (roi_c['h'] // 2)
    umbral_altura_minima = 0.1

    
    ultimo_y    = None
    ultimo_t    = None
    v_px_s      = 0.0
    v_m_s       = 0.0
    altura_disp = 0.0  

    memoria = {
        "clase": "ESPERANDO", "estado": "---", "vel": 0.0,
        "pos_y": 0, "t": time.time(), "h_aros": [0, 0, 0, 0], "delta": 0,
        "evaluado": False, "h_max": 0.0, "std_dev": 0.0, "color": (255, 255, 255)
    }

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f:
                continue

            color_img = np.asanyarray(color_f.get_data())
            depth_cm  = np.asanyarray(depth_f.get_data()) * depth_f.get_units() * 100
            alturas   = np.where((depth_cm > 0), DIST_REFERENCIA - depth_cm, 0)

            # Búsqueda de objetos en roi_v 
            region_alt = alturas[roi_v['y']:roi_v['y']+roi_v['h'],
                                 roi_v['x']:roi_v['x']+roi_v['w']]
            mask_v = (region_alt > umbral_altura_minima).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hay_objeto = False

            for cnt in cnts:
                if cv2.contourArea(cnt) < 800:
                    continue

                hay_objeto = True

                # Centroide 
                M  = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"]) + roi_v['x']
                cy = int(M["m01"] / M["m00"]) + roi_v['y']

                # Bounding box 
                bx_l, by_l, bw, bh = cv2.boundingRect(cnt)
                bx = bx_l + roi_v['x']
                by = by_l + roi_v['y']

                # Velocidad
                tiempo_actual = frames.get_timestamp() / 1000.0
                if ultimo_y is not None and ultimo_t is not None:
                    dt = tiempo_actual - ultimo_t
                    if dt > 0:
                        dy      = abs(cy - ultimo_y)
                        v_px_s  = dy / dt
                        cm_por_px = altura_disp / bh if bh > 0 else 0
                        v_m_s   = (v_px_s * cm_por_px) / 100.0
                ultimo_y = cy
                ultimo_t = tiempo_actual

                # Clasificación
                if abs(cy - LINEA_DISPARO) < 5 and not memoria["evaluado"]:
                    crop_alt = alturas[by:by+bh, bx:bx+bw]
                    h_max    = np.percentile(crop_alt, 98)
                    clase    = "GRANDE" if h_max > 13 else "MEDIANA" if h_max > 7 else "CHICA"
                    altura_disp = h_max   # actualizar referencia de altura

                    centro_lata = (bw // 2, bh // 2)
                    radios  = [int(bw * 0.15), int(bw * 0.25), int(bw * 0.35), int(bw * 0.45)]
                    h_aros  = [obtener_datos_aro(crop_alt, centro_lata, r) for r in radios]

                    estado, color, std_dev = clasificar_lata_avanzado(clase, h_aros)

                    memoria.update({
                        "clase": clase, "estado": estado, "h_max": h_max,
                        "h_aros": h_aros, "delta": h_aros[3] - h_aros[0],
                        "color": color, "std_dev": std_dev,
                        "evaluado": True, "vel": v_m_s, "radios_px": radios
                    })
                    guardar_registro(memoria)
                    print(f"[OK] {clase} → {estado}  |  vel={v_m_s:.3f} m/s")

                cv2.rectangle(color_img, (bx, by), (bx+bw, by+bh), memoria["color"], 2)
                cv2.circle(color_img, (cx, cy), 4, (0, 0, 255), -1)

                tx = bx + bw + 8
                cv2.putText(color_img, f"{memoria['clase']}",
                            (tx, by), 1, 1.2, (255, 255, 255), 2)
                cv2.putText(color_img, f"{memoria['estado']}",
                            (tx, by + 25), 1, 1.0, memoria["color"], 2)
                cv2.putText(color_img, f"D:{memoria['delta']:.2f} STD:{memoria['std_dev']:.2f}",
                            (tx, by + 48), 1, 0.8, (0, 255, 255), 1)

                # Cálculo de los 4 aros
                if memoria["evaluado"] and "radios_px" in memoria:
                    for r in memoria["radios_px"]:
                        cv2.circle(color_img, (cx, cy), r, (255, 255, 0), 1)

                # Velocidad debajo del bbox
                info_vel = f"{v_px_s:.1f} px/s | {v_m_s:.3f} m/s"
                cv2.putText(color_img, info_vel, (bx, by + bh + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                memoria.update({"pos_y": cy, "t": time.time()})
                break   # sólo procesar el contorno más grande

            #  Reseteo 
            if not hay_objeto:
                memoria["evaluado"] = False
                memoria["clase"]    = "ESPERANDO"
                memoria["color"]    = (255, 255, 255)
                ultimo_y  = None
                ultimo_t  = None
                v_px_s    = 0.0
                v_m_s     = 0.0
                altura_disp = 0.0

           
            # ROI v
            cv2.rectangle(color_img,
                          (roi_v['x'], roi_v['y']),
                          (roi_v['x']+roi_v['w'], roi_v['y']+roi_v['h']),
                          (100, 100, 100), 1)
            # Línea 
            cv2.line(color_img,
                     (roi_c['x'], LINEA_DISPARO),
                     (roi_c['x'] + roi_c['w'], LINEA_DISPARO),
                     (0, 0, 255), 2)

            # Panel lateral 
            dibujar_panel(color_img, v_px_s, v_m_s, memoria)

            cv2.imshow('Sistema Integrado — Velocidad + Calidad', color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()