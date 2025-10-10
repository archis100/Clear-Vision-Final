import os
import random
import cv2
import numpy as np  # <-- Add this
from tqdm import tqdm

# --- Define your degradation functions ---
def add_gaussian_noise(image, mean=0, std=10):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_motion_blur(image, kernel_size=5):
    """Add simple motion blur"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def add_occlusion(image, occ_size=50):
    """Add a random black rectangle"""
    h, w, _ = image.shape
    x1 = random.randint(0, w - occ_size)
    y1 = random.randint(0, h - occ_size)
    image[y1:y1+occ_size, x1:x1+occ_size] = 0
    return image

def jpeg_compression(image, quality=50):
    """Compress image and decompress to simulate JPEG artifacts"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

# --- Random degradation ---
def random_degradation(image, max_ops=3):
    operations = [add_gaussian_noise, add_motion_blur, add_occlusion, jpeg_compression]
    num_ops = random.randint(1, max_ops)
    chosen_ops = random.sample(operations, num_ops)
    
    for op in chosen_ops:
        image = op(image)
    
    return image

# --- Apply degradation to all images in a folder ---
def apply_corruption_to_folder(clean_dir, degraded_dir, size=(256, 256)):
    os.makedirs(degraded_dir, exist_ok=True)
    images = [img for img in os.listdir(clean_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in tqdm(images, desc="Applying Degradation"):
        img_path = os.path.join(clean_dir, filename)
        save_path = os.path.join(degraded_dir, filename)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, size)
        degraded = random_degradation(image)
        cv2.imwrite(save_path, degraded, [cv2.IMWRITE_JPEG_QUALITY, 100])

    return images

# --- For CLI testing ---
if __name__ == "__main__":
    clean_dir = 'static/uploads'
    degraded_dir = 'static/outputs/degraded'
    apply_corruption_to_folder(clean_dir, degraded_dir)
