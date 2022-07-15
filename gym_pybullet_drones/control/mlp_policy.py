# #
import torch
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.sac.policies import SACPolicy

class RPGMlpPolicy(torch.nn.Module):

    def __init__(self, save_variables, device):
        super(RPGMlpPolicy, self).__init__()
        self.policy = SACPolicy(**save_variables) 
        # self.policy =MlpPolicy(**save_variables)
        self.policy.actor = torch.nn.Sequential(self.policy.actor,
                                                     torch.nn.Tanh())
        self.device = device

    def load_weights(self, state_dict):
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.to(device=self.device)

    def forward(self, obs: torch.Tensor):
        action = self.policy._predict(obs, deterministic=True)
        return action
