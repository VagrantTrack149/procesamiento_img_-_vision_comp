import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

def limpiar_outliers(datos):
    if datos.size < 4: return np.mean(datos) if datos.size > 0 else 0
    q1, q3 = np.percentile(datos, [25, 75])
    iqr = q3 - q1
    filtro = datos[(datos >= q1 - 1.5 * iqr) & (datos <= q3 + 1.5 * iqr)]
    return np.mean(filtro) if filtro.size > 0 else np.mean(datos)

def escribir_registro(m):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = (
        f"{timestamp} | Clase: {m['clase']} | Estado: {m['estado']} | "
        f"H_Max: {m['h_max_mm']:.1f}mm | Borde: {m['h_borde_mm']:.1f}mm | "
        f"Centro: {m['h_centro_mm']:.1f}mm | Delta: {m['delta_mm']:.1f}mm | "
        f"Ratio: {m['ratio']:.3f}\n"
    )
    with open("datos_ajustados_abolladas_2.txt", "a") as f:
        f.write(linea)

def main():
    archivo_entrada = 'juntas_abolladas.bag'
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(archivo_entrada)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    DIST_REFERENCIA = 54.8  # cm

    roi_c = {'x': 270, 'y': 210, 'w': 180, 'h': 80}
    target_x = roi_c['x'] + (roi_c['w'] // 2)
    target_y = roi_c['y'] + (roi_c['h'] // 2)
    rango_px = 60

    lata_en_proceso = False
    memoria = {
        "clase": "ESPERANDO",
        "estado": "BUSCANDO",
        "color": (255, 255, 255)
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

            mask_roi = (alturas[roi_c['y']:roi_c['y']+roi_c['h'], roi_c['x']:roi_c['x']+roi_c['w']] > 0.5).astype(np.uint8) * 255
            contornos, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objeto_en_escena = False

            for cnt in contornos:
                if cv2.contourArea(cnt) > 1000:
                    objeto_en_escena = True
                    M = cv2.moments(cnt)
                    cx, cy = int(M["m10"]/M["m00"]) + roi_c['x'], int(M["m01"]/M["m00"]) + roi_c['y']

                    if not lata_en_proceso and abs(cx - target_x) < rango_px and abs(cy - target_y) < 3:
                        lata_en_proceso = True

                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        bx, by = bx + roi_c['x'], by + roi_c['y']
                        crop_alt = alturas[by:by+bh, bx:bx+bw]

                        # ALTURAS EN CM
                        h_max = np.percentile(crop_alt, 98)
                        clase = "GRANDE" if h_max > 13 else "MEDIANA" if h_max > 7 else "CHICA"

                        # BORDE
                        mask_anillo = np.zeros(crop_alt.shape, dtype=np.uint8)
                        cv2.circle(mask_anillo, (bw//2, bh//2), int(bw/2.1), 255, 2)
                        puntos_borde = crop_alt[mask_anillo == 255]
                        puntos_borde = puntos_borde[puntos_borde > (h_max * 0.7)]
                        h_borde = limpiar_outliers(puntos_borde)

                        # CENTRO
                        lx, ly = cx - bx, cy - by
                        h_centro = limpiar_outliers(crop_alt[ly-2:ly+2, lx-2:lx+2].flatten())

                        # MÉTRICAS
                        delta = abs(h_borde - h_centro)
                        ratio = h_borde / h_max if h_max > 0 else 0

                        # NUEVO CRITERIO
                        es_abollada = ratio < 0.85

                        # CONVERSIÓN A MM PARA GUARDAR
                        h_max_mm = h_max * 10
                        h_borde_mm = h_borde * 10
                        h_centro_mm = h_centro * 10
                        delta_mm = delta * 10

                        memoria.update({
                            "clase": clase,
                            "estado": "ABOLLADA" if es_abollada else "NORMAL",
                            "color": (0, 0, 255) if es_abollada else (0, 255, 0),

                            "h_max_mm": h_max_mm,
                            "h_borde_mm": h_borde_mm,
                            "h_centro_mm": h_centro_mm,
                            "delta_mm": delta_mm,
                            "ratio": ratio,

                            # PARA MOSTRAR EN CM
                            "h_max": h_max,
                            "h_borde": h_borde,
                            "h_centro": h_centro,
                            "delta": delta
                        })

                        escribir_registro(memoria)

            if not objeto_en_escena:
                lata_en_proceso = False

            # UI
            cv2.line(color_img, (target_x - rango_px, target_y), (target_x + rango_px, target_y), (255, 0, 255), 2)

            y0 = 30
            dy = 30

            cv2.putText(color_img, f"{memoria['clase']} | {memoria['estado']}", (30, y0), 1, 1.5, memoria['color'], 2)
            cv2.putText(color_img, f"Hmax: {memoria.get('h_max',0):.2f} cm", (30, y0+dy), 1, 1.2, (255,255,255), 2)
            cv2.putText(color_img, f"Borde: {memoria.get('h_borde',0):.2f} cm", (30, y0+2*dy), 1, 1.2, (255,255,255), 2)
            cv2.putText(color_img, f"Centro: {memoria.get('h_centro',0):.2f} cm", (30, y0+3*dy), 1, 1.2, (255,255,255), 2)
            cv2.putText(color_img, f"Delta: {memoria.get('delta',0):.2f} cm", (30, y0+4*dy), 1, 1.2, (255,255,255), 2)
            cv2.putText(color_img, f"Ratio: {memoria.get('ratio',0):.3f}", (30, y0+5*dy), 1, 1.2, (0,255,255), 2)

            cv2.imshow('Inspeccion de Frame Unico', color_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()