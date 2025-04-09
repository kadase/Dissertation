from abc import ABC, abstractmethod
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.environment import OnlineAttackEnv

class Attack(ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config

    @abstractmethod
    def generate(self, input_data):
        pass

    @abstractmethod
    def apply(self, input_data):
        pass

class RLAttack(Attack):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.env = OnlineAttackEnv(target_model=model, demo_mode=False)
        self.vec_env = DummyVecEnv([lambda: self.env])
        self.agent = PPO.load(config["agent_path"])
        self.device = config["device"]
        self.target_model = model
        self.perturbation_scale = config.get("perturbation_scale", 0.05)

    def generate(self, input_data):
        original_env = self.env.unwrapped
        obs = self.vec_env.reset()
        action, _ = self.agent.predict(obs)

        # Масштаб + ограничение амплитуды
        action = action * self.perturbation_scale
        action = np.clip(action, -0.1, 0.1)

        perturbed_obs = original_env.apply_perturbation(action[0])
        perturbed_obs = np.clip(perturbed_obs, 0, 1)

        perturbed_obs_tensor = torch.tensor(perturbed_obs, dtype=torch.float32).unsqueeze(0)
        if perturbed_obs_tensor.dim() == 4 and perturbed_obs_tensor.shape[-1] == 3:
            perturbed_obs_tensor = perturbed_obs_tensor.permute(0, 3, 1, 2).contiguous()

        perturbed_obs_tensor = torch.clamp(perturbed_obs_tensor, 0.0, 1.0)

        with torch.no_grad():
            original_output = self.target_model(input_data)
            perturbed_output = self.target_model(perturbed_obs_tensor)

        return perturbed_obs_tensor, torch.norm(original_output - perturbed_output)

    def apply(self, input_data):
        perturbed_obs, difference = self.generate(input_data)
        print(f"Difference in model predictions: {difference.item()}")
        return perturbed_obs
    
class FGSMAttack(Attack):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.epsilon = config.get("epsilon", 0.03)
        self.targeted = config.get("targeted", False)  # Включить целевые атаки
        self.target_class = config.get("target_class", None)  # Целевой класс

    def generate(self, input_data):
        input_tensor = input_data.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        
        if self.targeted:
            # Целевая атака: минимизируем вероятность правильного класса
            target = torch.tensor([self.target_class], dtype=torch.long).to(input_data.device)
            loss = torch.nn.functional.cross_entropy(output, target)
        else:
            # Нетаргетная атака: максимизируем потерю
            loss = torch.nn.functional.cross_entropy(output, torch.argmax(output, dim=1))
        
        loss.backward()
        perturbation = self.epsilon * torch.sign(input_tensor.grad)
        perturbed_data = input_tensor + perturbation
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data, perturbation

    def apply(self, input_data):
        perturbed_data, _ = self.generate(input_data)
        return perturbed_data
    
class PGDAttack(Attack):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.epsilon = config.get("epsilon", 0.03)
        self.alpha = config.get("alpha", 0.01)
        self.num_steps = config.get("num_steps", 10)
        self.targeted = config.get("targeted", False)  # Включить целевые атаки
        self.target_class = config.get("target_class", None)  # Целевой класс

    def generate(self, input_data):
        perturbed_data = input_data.clone().detach()
        
        for _ in range(self.num_steps):
            perturbed_data.requires_grad_(True)
            output = self.model(perturbed_data)
            
            if self.targeted:
                # Целевая атака: минимизируем вероятность правильного класса
                target = torch.tensor([self.target_class], dtype=torch.long).to(input_data.device)
                loss = torch.nn.functional.cross_entropy(output, target)
            else:
                # Нетаргетная атака: максимизируем потерю
                loss = torch.nn.functional.cross_entropy(output, torch.argmax(output, dim=1))
            
            loss.backward()
            perturbation = self.alpha * torch.sign(perturbed_data.grad)
            perturbed_data = perturbed_data + perturbation
            perturbed_data = torch.clamp(perturbed_data, input_data - self.epsilon, input_data + self.epsilon)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            perturbed_data = perturbed_data.detach()
        
        return perturbed_data, None

    def apply(self, input_data):
        perturbed_data, _ = self.generate(input_data)
        return perturbed_data

    def apply(self, input_data):
        perturbed_data, _ = self.generate(input_data)
        return perturbed_data
    
class CWAttack(Attack):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.confidence = config.get("confidence", 0)  # Параметр уверенности
        self.lr = config.get("lr", 0.01)  # Скорость обучения
        self.num_steps = config.get("num_steps", 100)  # Количество итераций

    def generate(self, input_data):
        # Инициализируем возмущения
        perturbed_data = input_data.clone().detach().requires_grad_(True)
        
        # Оптимизатор для обновления возмущений
        optimizer = torch.optim.Adam([perturbed_data], lr=self.lr)
        
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            
            # Вычисляем предсказания модели
            output = self.model(perturbed_data)
            
            # Вычисляем функцию потерь Carlini-Wagner
            correct_class = torch.argmax(output, dim=1)
            target_loss = output[0, correct_class] - torch.max(output[0, output[0] != correct_class])
            loss = torch.clamp(target_loss + self.confidence, min=0)
            
            # Добавляем регуляризацию по норме возмущения
            perturbation_norm = torch.norm(perturbed_data - input_data)
            loss += perturbation_norm
            
            # Обновляем возмущения
            loss.backward(retain_graph=True)
            optimizer.step()
            
            perturbed_data.data = torch.clamp(perturbed_data.data, 0, 1)
        
        return perturbed_data.detach(), None

    def apply(self, input_data):
        perturbed_data, _ = self.generate(input_data)
        return perturbed_data