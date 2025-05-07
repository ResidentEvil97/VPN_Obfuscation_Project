# test_env.py

from rl.vpn_env import VPNObfuscationEnv
import numpy as np

# Create environment instance
env = VPNObfuscationEnv(dpi_strategy="basic")

# Reset environment to get initial state
obs = env.reset()
total_reward = 0
for t in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}: reward={reward}, detected={info['detected']}")
    total_reward += reward
    if terminated or truncated:
        break
print("Total episode reward:", total_reward)


print("Initial observation:", obs)

# Take one random action from the action space
action = env.action_space.sample()
print("Sampled action:", action)

# Step through the environment
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
print("Next observation:", next_obs)
print("Reward:", reward)
print("Done?", done)
print("Info:", info)
