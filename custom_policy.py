from stable_baselines3.common.policies import ActorCriticCnnPolicy

class CustomCNNActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[dict(pi=[256, 256], vf=[256, 256])])