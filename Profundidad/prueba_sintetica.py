import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ==========================================
# 1. CONFIGURACIÓN Y PARÁMETROS INICIALES
# ==========================================
CONFIG = {
    "DIST_REFERENCIA": 54.8,
    "ROI_V": {'x': 245, 'y': 30, 'w': 200, 'h': 250},
    "LINEA_DISPARO": 240,
}

# Rangos para la optimización automática (Grid Search)
RANGO_DELTA = np.arange(-0.45, -0.10, 0.05)
RANGO_STD = np.arange(0.10, 0.25, 0.03)

def clasificar_lata(clase, h_aros, u_delta, u_std):
    """Función de clasificación con parámetros variables para optimización"""
    if len(h_aros) < 4 or any(h == 0 for h in h_aros):
        return "INCONCLUSO"

    a0, a1, a2, a3 = h_aros
    delta = a3 - a0 
    std_dev = np.std(h_aros)
    
    # Aplicamos los umbrales que estamos probando
    fail_delta = delta < u_delta
    fail_varianza = std_dev > u_std

    if fail_delta or fail_varianza:
        return "ABOLLADA"
    return "NORMAL"

# ==========================================
# 2. MOTOR DE PROCESAMIENTO DE .BAG
# ==========================================
def procesar_lote_bags(lista_archivos, u_delta, u_std):
    stats = {"aciertos": 0, "total": 0}
    
    for nombre_archivo in lista_archivos:
        if not os.path.exists(nombre_archivo):
            continue
            
        es_abollada_real = "abollada" in nombre_archivo.lower()
        
        pipeline = rs.pipeline()
        config = rs.config()
        # repeat_playback=False asegura que pase al siguiente archivo al terminar
        config.enable_device_from_file(nombre_archivo, repeat_playback=False)
        
        try:
            profile = pipeline.start(config)
            playback = profile.get_device().as_playback()
            playback.set_real_time(False) # Procesar a máxima velocidad
            align = rs.align(rs.stream.color)
            
            evaluado_en_este_frame = False
            
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    break # Fin del archivo .bag detectado

                aligned = align.process(frames)
                depth_f = aligned.get_depth_frame()
                if not depth_f: continue

                # Procesamiento de matriz de profundidad
                depth_cm = np.asanyarray(depth_f.get_data()) * depth_f.get_units() * 100
                alturas = np.where((depth_cm > 0), CONFIG["DIST_REFERENCIA"] - depth_cm, 0)

                roi = CONFIG["ROI_V"]
                mask = (alturas[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']] > 0.5).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(cnts) > 0:
                    for cnt in cnts:
                        if cv2.contourArea(cnt) > 800:
                            M = cv2.moments(cnt)
                            cy = int(M["m01"]/M["m00"]) + roi['y']
                            
                            # Disparo de lógica en línea crítica
                            if abs(cy - CONFIG["LINEA_DISPARO"]) < 5 and not evaluado_en_este_frame:
                                bx, by, bw, bh = cv2.boundingRect(cnt)
                                bx, by = bx + roi['x'], by + roi['y']
                                crop = alturas[by:by+bh, bx:bx+bw]
                                
                                # Simulación de muestreo por aros
                                radios = [int(bw*0.15), int(bw*0.25), int(bw*0.35), int(bw*0.45)]
                                h_aros = [np.mean(crop[int(bh//2 - r):int(bh//2 + r), int(bw//2 - r):int(bw//2 + r)]) for r in radios]

                                res = clasificar_lata("GENERICA", h_aros, u_delta, u_std)
                                
                                # Contabilizar éxito
                                es_correcto = (res == "ABOLLADA") if es_abollada_real else (res == "NORMAL")
                                stats["total"] += 1
                                if es_correcto: stats["aciertos"] += 1
                                evaluado_en_este_frame = True
                else:
                    evaluado_en_este_frame = False
        finally:
            pipeline.stop()
            
    return stats["aciertos"] / stats["total"] if stats["total"] > 0 else 0

# ==========================================
# 3. EJECUCIÓN PRINCIPAL Y OPTIMIZACIÓN
# ==========================================
if __name__ == "__main__":
    archivos = [
        'espaciadas_abolladas.bag', 'espaciadas_ordenada.bag', 
        'espaciadas_repetidas.bag', 'juntas_abolladas.bag', 
        'juntas_ordenadas.bag', 'juntas_repetidas.bag'
    ]

    print("Iniciando búsqueda de parámetros óptimos...")
    mejor_precision = 0
    mejor_config = {}

    # Búsqueda de rejilla para encontrar el punto dulce
    for d in RANGO_DELTA:
        for s in RANGO_STD:
            precision = procesar_lote_bags(archivos, d, s)
            print(f"Probando Delta: {d:.2f}, STD: {s:.2f} -> Precisión: {precision*100:.2f}%")
            
            if precision > mejor_precision:
                mejor_precision = precision
                mejor_config = {"delta": d, "std": s}

    print("\n" + "="*30)
    print("CONFIGURACIÓN ÓPTIMA HALLADA")
    print(f"Precisión Máxima: {mejor_precision*100:.2f}%")
    print(f"Parámetros: {mejor_config}")
    print("="*30)