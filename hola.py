import cv2 as cv
import numpy as np

#leer la imagen
img= cv.imread("./images/carreteraGrises.png")
print(img.shape)
cv.imshow("Imagen Original", img)

#convertir a escala de grises, quitar la capa del color y manejarlo por intensidad
img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)
cv.imshow("Imagen en Escala de Grises", img)

#Imagen binaria
img1=cv.imread("./images/bananaBinario.png")
print(img1.shape)
cv.imshow("Imagen Binaria Original", img1)
#convertir a escala de grises
img1=cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
print(img1.shape)
cv.imshow("Imagen en Escala de Grises", img1)


#leer imagen a color
img2=cv.imread("./images/flowers.jpg")
print(img2.shape)
cv.imshow("Imagen Original", img2)

#mostrar el contenido de cada capa de color
b, g, r= cv.split(img2)
cv.imshow("Canal Azul", b)
cv.imshow("Canal Verde", g)
cv.imshow("Canal Rojo", r)

#borrando un area de color de img2
img2_borrada=img2.copy()
img2_borrada[:, 100:200, :]=0 #borrando el area de columnas 100 a 200
cv.imshow("Imagen sin Area Especifica", img2_borrada)

#regiones de interes
imgROI= img2[100:300, 200:400]
#imgROI= img2[150:300, 20:210]

cv.imshow("Region de Interes", imgROI)

#redimensionar imagen
img_resized= cv.resize(img2,(500,500))
print(img_resized.shape)
cv.imshow("Imagen Redimensionada", img_resized)

#rotaciones de imagenes
img_rotated= cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imshow("Imagen rotada", img_rotated)
print(img_rotated.shape)

#rotacion por angulos para cualquier angulo
M=cv.getRotationMatrix2D((img1.shape[1]//2, img1.shape[0]//2), 45, 1) #matriz de rotacion
img_rotated2= cv.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
cv.imshow("Imagen rotada 2", img_rotated2)
print(img_rotated2.shape)




#traslasiones
rows, cols= img2.shape[:2]
M= cv.getRotationMatrix2D((cols/2, rows/2), 0, 1) #matriz de traslacion
M[0,2]=100 #traslacion en x
M[1,2]=50  #traslacion en y
img_traslada= cv.warpAffine(img2, M, (cols, rows))
cv.imshow("Imagen Trasladada", img_traslada)
print(img_traslada.shape)




#negativo de imagen
img3=cv.imread("./images/EchoCardiogram.png")
print(img3.shape)
cv.imshow("Imagen Original", img3)
img_negativo= 255-img3
cv.imshow("Imagen Negativa", img_negativo)
print(img_negativo.shape)

#negativo de una imagen a color
img_negativo_color= cv.bitwise_not(img2)
cv.imshow("Imagen Negativa a Color", img_negativo_color)
print(img_negativo_color.shape)

#gamma correction
img4=cv.imread("./images/Image100.png")
print(img4.shape)
cv.imshow("Imagen Original", img4)
def gamma_correction(image, gamma):
    invGamma= 1/gamma
    table= [((i/255.0) ** invGamma) * 255 for i in range(256)]
    table= np.array(table, np.uint8)
    return cv.LUT(image, table)
img_gamma_corrected= gamma_correction(img4, 2.2)
cv.imshow("Imagen con Correccion Gamma", img_gamma_corrected)
print(img_gamma_corrected.shape)

#histograma de una imagen
histogram= cv.calcHist([img4], [0], None, [256], [0,256])
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(histogram)
plt.show()

#histograma a color
colors= ('b','g','r')
for i, color in enumerate(colors):
    histogram= cv.calcHist([img4], [i], None, [256], [0,256])
    plt.plot(histogram, color=color)
    plt.xlim([0,256])
plt.show()







cv.waitKey(0)
cv.destroyAllWindows()
