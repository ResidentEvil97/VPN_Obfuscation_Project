"""
SAC-MAML VPN Traffic Obfuscation Trainer
=========================================

This script trains an SAC agent to play the VPN traffic obfuscation game.
It sets up the environment, checks API compliance, wraps the environment with a Monitor,
and defines the policy architecture. It then creates an SAC model instance
and passes it the environment, policy architecture, and hyperparameters.
It trains the model using the learn method and saves it to a file.

"""

# Import VPNObfuscationEnv, SAC, Monitor, and env_checker from stable_baselines3
from rl.vpn_env import VPNObfuscationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
# Import ProgressBarCallback for displaying training progress
from stable_baselines3.common.callbacks import ProgressBarCallback
# Import os for creating directories
import os


# Set up the environment
# Create a VPNObfuscationEnv with a simple DPI strategy for initial training
# env = VPNObfuscationEnv(dpi_strategy="basic")

# Create a VPNObfuscationEnv with a machine learning-based DPI strategy
env = VPNObfuscationEnv(dpi_strategy="ml")

# Optional: check API compliance
# Use env_checker to check the environment for API compliance
check_env(env, warn=True)

# Wrap the environment with a Monitor to track training metrics
# Create a Monitor instance and pass it the environment
env = Monitor(env)

# Create a directory for logs and saved models
# Create a directory for logs
log_dir = "./logs/"
# Create a directory for saved models
model_dir = "./models/"
# Create the directories if they don't already exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set up the SAC model
# Define the policy architecture
policy_kwargs = dict(net_arch=[256, 256, 128])
# Create an SAC model instance and pass it the environment, policy architecture, and hyperparameters
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
# Train the model for 10000 timesteps
timesteps = 10000
print(f"Training SAC model for {timesteps} timesteps...")
# Train the model using the learn method
model.learn(total_timesteps=timesteps)

# Train the model using the learn method with a ProgressBarCallback
model.learn(
    total_timesteps=timesteps,
    callback=ProgressBarCallback()
)

# Save the model
# Save the model to a file
model_path = os.path.join(model_dir, "sac_vpn_obfuscation_ml")
model.save(model_path)
# Print a message indicating that the model has been saved
print(f"Model saved to {model_path}.zip")
