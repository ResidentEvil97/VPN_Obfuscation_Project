# Evaluate model performance across different DPI strategies

# Config
# List of DPI strategies to evaluate
strategies = ["basic", "strict", "noisy", "ml"]
# Number of episodes to evaluate per strategy
episodes = 10
# Number of steps per episode
steps_per_episode = 100
# Path to the trained model
model_path = "models/sac_vpn_obfuscation_ml.zip"

# Load Trained Model
# Load the trained SAC model
model = SAC.load(model_path)

# Storage for Evaluation Results
# List to store average rewards per strategy
rewards = []
# List to store detection rates per strategy
detection_rates = []

# Evaluate Model Across DPI Strategies
# Iterate over the list of DPI strategies
for strategy in strategies:
    # Create an environment with the current DPI strategy
    env = VPNObfuscationEnv(dpi_strategy=strategy)
    # Initialize total reward and detection count
    total_reward = 0
    detections = 0

    # Run episodes
    for _ in range(episodes):
        # Reset environment
        obs, _ = env.reset()
        # Run steps
        for _ in range(steps_per_episode):
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            # Take action in the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            # Update total reward
            total_reward += reward
            # Update detection count
            detections += obs[2]  # detection_flag in env = 1.0 if detected

    # Calculate average reward
    avg_reward = total_reward / (episodes * steps_per_episode)
    # Calculate detection rate
    detection_rate = detections / (episodes * steps_per_episode)

    # Store results
    rewards.append(avg_reward)
    detection_rates.append(detection_rate)

    # Print results
    print(f"{strategy.upper()} DPI — Avg Reward: {avg_reward:.3f}, Detection Rate: {detection_rate:.2%}")

# --------------------------------------------------------------------------------------------------
# Plot Results
# Create a figure with a single axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar chart for average reward
# Create bars for average rewards
bar = ax1.bar(strategies, rewards, color='green', alpha=0.6, label='Avg Reward')
# Set y-axis label
ax1.set_ylabel("Average Reward")
# Set y-axis limits
ax1.set_ylim(min(rewards) - 0.2, 1.1)

# Add value labels on top of bars
# Iterate over bars and add value labels
for rect, value in zip(bar, rewards):
    # Get height of bar
    height = rect.get_height()
    # Add value label
    ax1.text(rect.get_x() + rect.get_width() / 2.0, height + 0.05, f"{value:.2f}", ha='center', va='bottom')

# Line chart for detection rate
# Create a second axis
ax2 = ax1.twinx()
# Plot detection rates
ax2.plot(strategies, detection_rates, 'ro-', label='Detection Rate', linewidth=2)
# Set y-axis label
ax2.set_ylabel("Detection Rate")
# Set y-axis limits
ax2.set_ylim(0, 1.0)

# Add value labels to points
# Iterate over points and add value labels
for x, y in zip(strategies, detection_rates):
    # Add value label
    ax2.text(x, y + 0.05, f"{y:.2%}", ha='center', va='bottom', color='red')

# Set title
plt.title("Model Performance Across DPI Strategies", fontsize=14)
# Add grid
ax1.grid(True, which='both', linestyle='--', alpha=0.3)
# Adjust layout
fig.tight_layout()

# Save and show
plt.savefig("evaluation_results.png")
plt.show()
