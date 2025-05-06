import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from rl.vpn_env import VPNObfuscationEnv

# Config
strategies = ["basic", "strict", "noisy"]
episodes = 10
steps_per_episode = 100
model_path = "models/sac_vpn_obfuscation"

# Load Trained Model
model = SAC.load(model_path)

# Storage for Evaluation Results
rewards = []
detection_rates = []

# Evaluate Model Across DPI Strategies
for strategy in strategies:
    env = VPNObfuscationEnv(dpi_strategy=strategy)
    total_reward = 0
    detections = 0

    for _ in range(episodes):
        obs = env.reset()
        for _ in range(steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _ = env.step(action)
            total_reward += reward
            detections += obs[2]  # detection_flag in env = 1.0 if detected

    avg_reward = total_reward / (episodes * steps_per_episode)
    detection_rate = detections / (episodes * steps_per_episode)

    rewards.append(avg_reward)
    detection_rates.append(detection_rate)

    print(f"{strategy.upper()} DPI — Avg Reward: {avg_reward:.3f}, Detection Rate: {detection_rate:.2%}")

# --------------------------------------------------------------------------------------------------
# Plot Results
fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar chart for rewards
ax1.bar(strategies, rewards, color='green', alpha=0.6, label='Avg Reward')
ax1.set_ylabel("Average Reward")
ax1.set_ylim(-2, 2)

# Line plot for detection rate
ax2 = ax1.twinx()
ax2.plot(strategies, detection_rates, 'ro-', label='Detection Rate')
ax2.set_ylabel("Detection Rate")
ax2.set_ylim(0, 1)

plt.title("Model Performance Across DPI Strategies")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_results.png")
plt.show()
