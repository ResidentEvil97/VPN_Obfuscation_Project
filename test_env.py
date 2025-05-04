# test_env.py

from rl.vpn_env import VPNObfuscationEnv
import numpy as np

# Create environment instance
env = VPNObfuscationEnv(dpi_strategy="basic")

# Reset environment to get initial state
obs = env.reset()
print("Initial observation:", obs)

# Take one random action from the action space
action = env.action_space.sample()
print("Sampled action:", action)

# Step through the environment
next_obs, reward, done, info = env.step(action)
print("Next observation:", next_obs)
print("Reward:", reward)
print("Done?", done)
print("Info:", info)
