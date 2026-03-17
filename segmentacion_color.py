import cv2 as cv
import numpy as np

# Variables globales para almacenar los clicks
clicks = []
colores_rgb = []
img_original = None



def click_event(event, x, y, flags, params):
    global img_original, clicks, colores_rgb
    
    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicks) < 2:
            clicks.append((x, y))

            b, g, r = img_original[y, x]
            colores_rgb.append((r, g, b))
            print(f'{r},{g},{b}  ')
            
            img_marca = img_original.copy()
            cv.circle(img_marca, (x, y), 5, (0, 255, 0), -1)
            cv.imshow('original', img_marca)
            
            if len(clicks) == 2:
                segmentar_imagen()

def segmentar_imagen():
    
    global img_original, clicks, colores_rgb
    
    h, w, ch = img_original.shape
    
    color1_rgb = colores_rgb[0]
    color2_rgb = colores_rgb[1]
    
    print(f"Color 1{color1_rgb}")
    print(f"Color 2 {color2_rgb}")
    
    img_rgb = cv.cvtColor(img_original, cv.COLOR_BGR2RGB).astype(np.float32)
    
    # Convertir a amtriz 3x1
    color1_rgb = np.array(color1_rgb, dtype=np.float32)
    color2_rgb = np.array(color2_rgb, dtype=np.float32)
    
    pixels = img_rgb.reshape(-1, 3)  # Cada píxel es un punto en 3D (R, G, B)
    
    distancia1 = np.linalg.norm(pixels - color1_rgb, axis=1)
    distancia1 = distancia1.reshape(h, w)
    
    distancia2 = np.linalg.norm(pixels - color2_rgb, axis=1)
    distancia2 = distancia2.reshape(h, w)
    
    distancia1_norm = cv.normalize(distancia1, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    distancia2_norm = cv.normalize(distancia2, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # Crear máscara: región cercana a color1 vs región cercana a color2
    mascara_color1 = (distancia1 < distancia2).astype(np.uint8) * 255
    mascara_color2 = (distancia2 < distancia1).astype(np.uint8) * 255
    
    # Crear imagen segmentada con colores
    img_segmentada = np.zeros_like(img_original)
    #img_segmentada[mascara_color1 == 1] = [0, 0, 255]  # Rojo 
    #img_segmentada[mascara_color2 == 1] = [255, 0, 0]  # Azul 
    
    #img_segmentada=cv.bitwise_xor(img_original,mascara_color1)
    
    cv.imshow('distancia ', distancia1_norm)
    cv.imshow('distancia ', distancia2_norm)
    cv.imshow('Color 1', mascara_color1)
    cv.imshow('Color 2', mascara_color2)
    cv.imshow('Imagen Segmentada', img_segmentada)
    
if __name__=="__main__":
    img_original=cv.imread('images/windex.bmp')
    
    h, w, ch = img_original.shape
    print(f"{w}x{h}")

        
    cv.imshow('original', img_original)
    cv.setMouseCallback('original', click_event)
        
    cv.waitKey(0)
    cv.destroyAllWindows()