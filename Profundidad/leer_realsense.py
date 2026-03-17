import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    archivo_entrada = 'grabacion_20260312_190040.bag'

    # Configurar los flujos (streams) de la cámara RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    # Indicar a RealSense que lea desde el archivo .bag en lugar de una cámara física
    config.enable_device_from_file(archivo_entrada)

    # Habilitar los flujos de forma automática basándose en lo que hay en el .bag
    # Al no forzar 'enable_stream' con FPS o resolución fijas, se adaptará al video.

    print(f"[INFO] Iniciando pipeline de RealSense para leer '{archivo_entrada}'...")

    # Iniciar el pipeline
    profile = pipeline.start(config)

    # (Opcional) Evitar que el video se reproduzca en bucle (repetir indefinidamente)
    playback = profile.get_device().as_playback()
    playback.set_real_time(True) # True = Reproducción en tiempo real, False = lo más rápido posible

    try:
        print("[INFO] Reproduciendo... Presiona 'q' o la tecla 'Esc' en la ventana de video para salir.")
        while True:
            # Esperar hasta que lleguen los frames (lanzará un error si el archivo termina y no hace loop)
            try:
                # Utilizamos un timeout alto por si el disco es un poco lento leyendo
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                # Si wait_for_frames lanza un timeout, puede que haya terminado la grabación
                print("[INFO] Fin del archivo alcanzado o no hay más frames.")
                break

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convertir los frames a arreglos de NumPy para poder visualizarlos con OpenCV
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # La imagen de profundidad es de 16 bits; aplicamos un mapa de colores
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.365), cv2.COLORMAP_JET)

            # Juntar la imagen RGB y el mapa de profundidad horizontalmente
            images = np.hstack((color_image, depth_colormap))

            # Mostrar las imágenes combinadas en una ventana
            cv2.namedWindow('Reproduccion .bag - RGB + Profundidad', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Reproduccion .bag - RGB + Profundidad', images)

            # Como waitKey detiene la ejecución ese tiempo, ayuda a que OpenCV pinte el cuadro. 
            # El pipeline de RealSense ya maneja el tiempo real (set_real_time=True).
            tecla = cv2.waitKey(1)
            if tecla & 0xFF == ord('q') or tecla == 27:
                break

    finally:
        # Detener la reproducción de forma segura al salir
        print("[INFO] Deteniendo la reproducción y limpiando...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
