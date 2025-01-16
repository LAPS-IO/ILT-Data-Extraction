import os
import sys
import cv2
import numpy as np

# Read the image
img_path = os.path.abspath(sys.argv[1])
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Compute the Fourier Transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Set a cutoff frequency
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
size = 25
mask[crow - size:crow + size, ccol - size:ccol + size] = 1

# Apply the filter
dft_shift_filtered = dft_shift * mask

# Compute the inverse Fourier Transform
dft_filtered = np.fft.ifftshift(dft_shift_filtered)
img_back = cv2.idft(dft_filtered)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize and display the image
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('Image', img_back.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
