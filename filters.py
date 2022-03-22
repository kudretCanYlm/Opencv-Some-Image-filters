# Median Filter

# Generate random integers from 0 to 20


from matplotlib.pyplot import gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

img = cv2.imread('monalisa.jfif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
mean = 0
var = 150
sigma = var*0.4
row, col = 274, 184

np.random.seed(0)
gray_sp = gray*1
sp_indices = np.random.randint(0, 21, [row, col])

for i in range(row):
    for j in range(col):
        if sp_indices[i, j] == 0:
            gray_sp[i, j] = 0
        if sp_indices[i, j] == 20:
            gray_sp[i, j] = 255
plt.imshow(gray_sp, cmap="gray")
plt.show()

# Now we want to remove the salt and pepper noise through a Median filter.
# Using the opencv Median filter for the same

gray_sp_removed = cv2.medianBlur(gray_sp, 3)
plt.imshow(gray_sp_removed, cmap="gray")
plt.show()

# Implementation of the 3x3 Median filter without using opencv

gray_sp_removed_exp = gray*1
for i in range(row):
    for j in range(col):
        local_arr = []
        for k in range(np.max([0, i-1]), np.min([i+2, row])):
            for l in range(np.max([0, j-1]), np.min([j+2, col])):
                local_arr.append(gray_sp[k, l])
        gray_sp_removed_exp[i, j] = np.median(local_arr)
plt.imshow(gray_sp_removed_exp, cmap="gray")
plt.show()

# Gaussian Filter
Hg = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        Hg[i, j] = np.exp(-((i-10)**2+(j-10)**2)/10)
plt.imshow(Hg, cmap="gray")
plt.show()
gray_blur = convolve2d(gray, Hg, mode="same")
plt.imshow(gray-gray_blur, cmap="gray")
plt.show()
gray_enhanced = gray+0.025*(gray-gray_blur)
plt.imshow(gray_enhanced, cmap="gray")
plt.show()

# Gradient-based Filters
# vertical filter
gradient_filter_vertical = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0]])
vertical_filter_image = convolve2d(gray, gradient_filter_vertical, mode="same")

plt.imshow(vertical_filter_image, cmap="gray")
plt.show()

# horizontal filter
gradient_filter_horizontal = np.array([
    [0, 0, 0],
    [1, 0, -1],
    [0, 0, 0]])
horizontal_filter_image = convolve2d(
    gray, gradient_filter_horizontal, mode="same")
plt.imshow(horizontal_filter_image, cmap="gray")
plt.show()

# Sobel Edge-Detection Filter
Hx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]], dtype=np.float32)

Gx = convolve2d(gray, Hx, mode="same")
plt.imshow(Gx, cmap='gray')
plt.show()

Hy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]], dtype=np.float32)
Gy = convolve2d(gray, Hy, mode="same")
plt.imshow(Gy, cmap='gray')
plt.show()

G = (Gx*Gx+Gy*Gy)**0.5
plt.imshow(G, cmap="gray")
plt.show()
