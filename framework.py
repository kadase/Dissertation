import os
import yaml
import cv2
import torch
import sys
from pathlib import Path
import numpy as np
import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core import ModelFactory, AttackFactory
from core.environment import OnlineAttackEnv
from core.utils import save_results
from collections import deque, Counter

def save_attack_metadata(original_class, adversarial_class, original_conf, adversarial_conf, perturbation, save_dir="logs/attack_meta"):
    import json
    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        "original_class": int(original_class),
        "adversarial_class": int(adversarial_class),
        "original_confidence": round(float(original_conf), 4),
        "adversarial_confidence": round(float(adversarial_conf), 4),
        "delta_prediction": round(abs(float(original_conf - adversarial_conf)), 4),
        "l2_distance": round(float(np.linalg.norm(perturbation.detach().cpu().numpy())), 4),
        "success": int(original_class) != int(adversarial_class)
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"attack_meta_{timestamp}.json")
    try:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
            f.flush()
        print(f"Метаинформация сохранена: {path}")
    except Exception as e:
        print(f"Ошибка при сохранении JSON: {e}")

def prepare_for_display(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.max() <= 1.0:
        image = image * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def preprocess(frame_rgb, data_config):
    resized = cv2.resize(frame_rgb, (84, 84))
    tensor = torch.tensor(resized).float()
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor

def main():
    config_path = project_root / "configs" / "attack_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = ModelFactory.create(config["model"])
    env = OnlineAttackEnv(target_model=model, demo_mode=False, use_target_model=True)
    attack = AttackFactory.create(config["attack"], model)

    cap = cv2.VideoCapture(0)
    prediction_history = deque(maxlen=5)  # Храним последние 5 предсказаний
    last_stable_pred = None
    if not cap.isOpened():
        raise RuntimeError("Camera initialization failed")

    last_pred = None  # <-- Инициализация переменной до начала цикла

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame capture error")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = preprocess(frame_rgb, config["data"])

        with torch.no_grad():
            if processed.dim() == 3:
                processed = processed.unsqueeze(0)
            original_output = model(processed)
            original_pred = original_output.argmax().item()
            
        # Добавим предсказание в историю
        prediction_history.append(original_pred)

        # Находим самое частое предсказание в истории
        most_common_pred, freq = Counter(prediction_history).most_common(1)[0]

        # Проверяем, изменился ли "стабильный" класс
        if last_stable_pred is None or most_common_pred != last_stable_pred:
            print(f"⚠️ Стабильное предсказание изменилось: {last_stable_pred} → {most_common_pred}")
        last_stable_pred = most_common_pred
        
        # Проверяем изменение предсказания
        if last_pred != original_pred:
            print(f"⚠️ Предсказание изменилось: {last_pred} → {original_pred}")
        last_pred = original_pred

        adversarial = attack.apply(processed)
        if adversarial.dim() == 3:
            adversarial = adversarial.unsqueeze(0)

        with torch.no_grad():
            perturbed_output = model(adversarial)
            perturbed_pred = perturbed_output.argmax().item()

        difference = torch.norm(original_output - perturbed_output).item()
        print(f"Difference in model predictions: {difference}")

        adv_np = adversarial.squeeze().detach().cpu().numpy()
        adv_np = np.transpose(adv_np, (1, 2, 0))
        adv_np = prepare_for_display(adv_np)
        adv_display_bgr = adv_np

        attack_type = config["attack"]["type"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_folder = os.path.join(project_root, "logs", attack_type)
        os.makedirs(attack_folder, exist_ok=True)

        save_path = os.path.join(attack_folder, f"attack_results_{attack_type}_{timestamp}.png")
        save_results(frame_rgb, adv_np, original_pred, perturbed_pred, save_path)

        perturbation = adversarial.squeeze() - processed.squeeze()
        original_prob = torch.nn.functional.softmax(original_output, dim=1)[0]
        perturbed_prob = torch.nn.functional.softmax(perturbed_output, dim=1)[0]
        orig_conf = original_prob[original_pred].item()
        perturbed_conf = perturbed_prob[perturbed_pred].item()

        save_attack_metadata(original_pred, perturbed_pred, orig_conf, perturbed_conf, perturbation)

        cv2.imshow('Original', frame)
        cv2.imshow('Adversarial', adv_display_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
