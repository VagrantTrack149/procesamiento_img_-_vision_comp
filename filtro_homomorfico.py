import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def homofilter(I):
    # show original image (convert BGR to RGB for display)
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(I, cv.COLOR_BGR2RGB))
    plt.title("Imagen original")
    plt.axis("off")

    # convert to grayscale and float
    gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY).astype(np.float64)
    m, n = gray.shape

    # filter parameters
    rL = 0.5
    rH = 2.0
    c = 2.0
    d0 = 20.0

    # log transform and FFT
    I1 = np.log(gray + 1)
    FI = np.fft.fft2(I1)

    # create distance squared matrix centered at zero frequency
    u = np.arange(m) - m // 2
    v = np.arange(n) - n // 2
    V, U = np.meshgrid(v, u)   # note order: columns, rows
    D2 = U**2 + V**2

    # homomorphic filter transfer function
    H = (rH - rL) * (1 - np.exp(-c * (D2 / (d0**2)))) + rL

    # apply filter and inverse FFT
    I2 = np.fft.ifft2(H * FI)
    I3 = np.real(np.exp(I2)) - 1

    # normalize result for display
    I3 = cv.normalize(I3, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    plt.subplot(1, 2, 2)
    plt.imshow(I3, cmap="gray")
    plt.title("imagen mejorada por filtrado homomórfico")
    plt.axis("off")
    plt.show()


# configure matplotlib to support Chinese characters if needed
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# example usage
img = cv.imread("./images/tunel.jpeg")
if img is None:
    print('Failed to load capture.png')
else:
    homofilter(img)

