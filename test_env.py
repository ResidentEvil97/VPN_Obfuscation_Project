# test_env.py

from rl.vpn_env import VPNObfuscationEnv
import numpy as np

# Create environment instance
env = VPNObfuscationEnv(dpi_strategy="basic")

# Reset environment to get initial state
obs = env.reset()
total_reward = 0
for t in range(200):
    # Take a random action
    action = env.action_space.sample()
    # Step through the environment
    obs, reward, terminated, truncated, info = env.step(action)
    # Print reward and detection status
    print(f"Step {t}: reward={reward}, detected={info['detected']}")
    total_reward += reward
    # Break if episode is done
    if terminated or truncated:
        break
# Print total episode reward
print("Total episode reward:", total_reward)

# Print initial observation
print("Initial observation:", obs)

# Take one random action from the action space
action = env.action_space.sample()
# Print sampled action
print("Sampled action:", action)

# Step through the environment
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
# Print next observation
print("Next observation:", next_obs)
# Print reward
print("Reward:", reward)
# Print done status
print("Done?", done)
# Print info
print("Info:", info)
