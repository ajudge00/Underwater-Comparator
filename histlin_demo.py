import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/lenna_low_contrast.jpg", cv2.IMREAD_GRAYSCALE)

# NORMAL HISTOGRAM EQ
hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
pdf = hist / np.sum(hist)
cdf = np.cumsum(pdf)
cdf_scaled = (cdf * 255).astype(np.uint8)
equalized_image = cdf_scaled[image]
equalized_hist, _ = np.histogram(equalized_image.flatten(), bins=256, range=[0, 256])

# HISTOGRAM "LINEARIZATION"
r = 0.4
P_max = np.max(pdf)

pdf_lind = np.where((pdf > 0) & (pdf < 255), (pdf / P_max) ** r * P_max, pdf)
pdf_lind = pdf_lind / np.sum(pdf_lind)
cdf_lind = np.cumsum(pdf_lind)
cdf_scaled_lind = (cdf_lind * 255).astype(np.uint8)
equalized_image_lind = cdf_scaled_lind[image]
equalized_hist_lind, _ = np.histogram(equalized_image_lind.flatten(), bins=256, range=[0, 256])


plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.bar(np.arange(256), hist, color="black")
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Count")

plt.subplot(3, 2, 3)
plt.imshow(equalized_image, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.bar(np.arange(256), equalized_hist, color="black")
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Count")

plt.subplot(3, 2, 5)
plt.imshow(equalized_image_lind, cmap="gray")
plt.title("Linearized Image")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.bar(np.arange(256), equalized_hist_lind, color="black")
plt.title("Linearized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print(equalized_image - equalized_image_lind)