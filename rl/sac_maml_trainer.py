# sac_maml_trainer.py

from rl.vpn_env import VPNObfuscationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback
import os

# Set up the environment

# Use a simple DPI strategy for initial training
# env = VPNObfuscationEnv(dpi_strategy="basic")
env = VPNObfuscationEnv(dpi_strategy="ml")


# Optional: check API compliance
check_env(env, warn=True)

# Wrap the environment with a Monitor to track training metrics
env = Monitor(env)

# Create a directory for logs and saved models
log_dir = "./logs/"
model_dir = "./models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set up the SAC model


policy_kwargs = dict(net_arch=[256, 256, 128])
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=500_000,
    batch_size=512,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto_0.1",
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs
)
# Train the model

timesteps = 10000
print(f"Training SAC model for {timesteps} timesteps...")
model.learn(total_timesteps=timesteps)

model.learn(
    total_timesteps=timesteps,
    callback=ProgressBarCallback()
)

# Save the model

model_path = os.path.join(model_dir, "sac_vpn_obfuscation_ml")
model.save(model_path)
print(f"Model saved to {model_path}.zip")
