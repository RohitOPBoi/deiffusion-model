import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import cv2

def load_img(path):
    img = Image.open(path).convert("L").resize((256, 256))
    return np.array(img)

original = load_img("refine_outputs/original.png")
refined  = load_img("refine_outputs/refined.png")

ssim_score = ssim(original, refined)
mse = np.mean((original - refined) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)

error_map = np.abs(original - refined)

error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min())

h, w = original.shape
crop = original[h//3:h//3+80, w//3:w//3+80]
crop_ref = refined[h//3:h//3+80, w//3:w//3+80]

fig = plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.imshow(original, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(refined, cmap='gray')
plt.title("Refined")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(error_norm, cmap='jet')
plt.title("Error Heatmap")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(crop, cmap='gray')
plt.title("Zoom Original")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(crop_ref, cmap='gray')
plt.title("Zoom Refined")
plt.axis('off')

plt.subplot(2,3,6)
plt.text(0.1, 0.6, f"SSIM: {ssim_score:.3f}", fontsize=12)
plt.text(0.1, 0.4, f"PSNR: {psnr:.2f} dB", fontsize=12)
plt.title("Metrics")
plt.axis('off')

plt.tight_layout()
plt.savefig("final_analysis.png", dpi=300)
plt.show()

print("Saved: final_analysis.png")
print("saved")
