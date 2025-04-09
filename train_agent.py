import os
import sys
import importlib
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

# Добавляем корень проекта в sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from core.custom_cnn import CustomCNN
from core.custom_policy import CustomCNNActorCriticPolicy
from core.callbacks import TrainingLogger
from stable_baselines3.common.callbacks import BaseCallback

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Загрузка train_config.yaml
with open(os.path.join(project_root, "configs", "train_config.yaml"), "r") as f:
    TRAIN_CONFIG = yaml.safe_load(f)

class TQDMProgressBar(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.pbar.update(self.locals["n_steps"])
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

def train_rl_agent():
    env = make_vec_env(
        "OnlineAttackEnv-v0",
        n_envs=TRAIN_CONFIG["hyperparams"]["n_envs"],
        vec_env_cls=DummyVecEnv
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    obs = env.reset()
    if np.isnan(obs).any() or np.isinf(obs).any():
        raise ValueError("Наблюдения содержат NaN или бесконечные значения!")

    print("Initializing model...")

    hyperparams = TRAIN_CONFIG["hyperparams"].copy()

    # Поддержка строкового пути до кастомного экстрактора
    if "policy_kwargs" in hyperparams:
        features_class_str = hyperparams["policy_kwargs"]["features_extractor_class"]
        module_name, class_name = features_class_str.rsplit(".", 1) if "." in features_class_str else ("core.custom_cnn", features_class_str)
        module = importlib.import_module(module_name)
        hyperparams["policy_kwargs"]["features_extractor_class"] = getattr(module, class_name)

    hyperparams["learning_rate"] = float(hyperparams["learning_rate"])
    policy = hyperparams.pop("policy_type", "CnnPolicy")
    total_timesteps = hyperparams.pop("total_timesteps", None)
    hyperparams.pop("n_envs", None)

    model = PPO(
        policy=policy,
        env=env,
        **hyperparams
    )

    print("Старт обучения...")
    logger = TrainingLogger(log_file=os.path.join(project_root, TRAIN_CONFIG["paths"]["logs"], "training_log.json"))

    try:
        progress = TQDMProgressBar(total_timesteps=total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=[logger, progress])   
    except ValueError as e:
        print(f"Ошибка во время обучения: {e}")
        raise

    print("Обучение выполнено! Сохранение модели...")
    model.save(os.path.join(project_root, TRAIN_CONFIG["paths"]["agents"], "rl_agent.zip"))

if __name__ == "__main__":
    train_rl_agent()
