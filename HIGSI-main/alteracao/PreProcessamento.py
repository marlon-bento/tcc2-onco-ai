import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path da Imagem
img_path = '1-231.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)

img_median = cv2.medianBlur(img_clahe, ksize=5)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))

axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original ')
axs[0].axis('off')

axs[1].imshow(img_clahe, cmap='gray')
axs[1].set_title('CLAHE aplicado')
axs[1].axis('off')

axs[2].imshow(img_median, cmap='gray')
axs[2].set_title('Filtro Mediano')
axs[2].axis('off')

plt.tight_layout()
plt.show()
