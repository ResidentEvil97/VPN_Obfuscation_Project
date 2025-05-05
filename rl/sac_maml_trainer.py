# sac_maml_trainer.py

import VPNObfuscationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# Optional: check that the environment conforms to Gym's API
env = VPNObfuscationEnv(dpi_strategy="basic")
check_env(env, warn=True)
