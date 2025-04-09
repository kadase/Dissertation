from .model import PyTorchModel  # Импортируем PyTorchModel
from .attacks import RLAttack, FGSMAttack, PGDAttack, CWAttack

class AttackFactory:
    @staticmethod
    def create(config, model):
        if config["type"] == "rl_attack":
            return RLAttack(model, config["params"])
        elif config["type"] == "fgsm":
            return FGSMAttack(model, config["params"])
        elif config["type"] == "pgd":
            return PGDAttack(model, config["params"])
        elif config["type"] == "cw":
            return CWAttack(model, config["params"])
        raise ValueError(f"Unknown attack type: {config['type']}")

class ModelFactory:
    @staticmethod
    def create(config):
        if config["type"] == "pytorch":
            return PyTorchModel(config["params"])
        raise ValueError(f"Unknown model type: {config['type']}")