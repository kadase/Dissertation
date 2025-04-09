import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

def visualize_attack(original_image, perturbed_image, original_pred, perturbed_pred):
    plt.figure(figsize=(10, 5))
    
    # Оригинальное изображение
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image\nPrediction: {original_pred}")
    plt.axis('off')
    
    # Возмущенное изображение
    plt.subplot(1, 2, 2)
    plt.imshow(perturbed_image)
    plt.title(f"Perturbed Image\nPrediction: {perturbed_pred}")
    plt.axis('off')
    
    plt.show()

import numpy as np

def save_results(original_image, perturbed_image, original_pred, perturbed_pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Приводим к uint8
    def to_uint8(img):
        if img.max() <= 1.0:
            img = img * 255
        return img.astype(np.uint8)

    original_image = to_uint8(original_image)
    perturbed_image = to_uint8(perturbed_image)

    desired_size = (224, 144)
    original_resized = cv2.resize(original_image, desired_size)
    adversarial_resized = cv2.resize(perturbed_image, desired_size)
    adversarial_resized = cv2.cvtColor(adversarial_resized, cv2.COLOR_RGB2BGR)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_resized)
    plt.title(f"Original Image\nPrediction: {original_pred}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_resized)
    plt.title(f"Perturbed Image\nPrediction: {perturbed_pred}")
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()


    # --- Предобработка ---
def preprocess(frame, config):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config["preprocess"]["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame)