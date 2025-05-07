import numpy as np
import os
import shutil
import copy
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from rl.vpn_env import VPNObfuscationEnv

# --- Hyperparameters ---
meta_batch_size = 2         # Number of tasks per meta-iteration
meta_iterations = 3        # Number of meta-iterations
inner_steps = 1             # Number of adaptation steps per task
adapt_steps_timesteps = 50 # Timesteps for each adaptation step
eval_episodes = 1           # Evaluation episodes per task
policy_kwargs = dict(net_arch=[64, 64])

# --- Utility: Create a randomized environment (TASK) ---
def make_random_env():
    dpi_strategy = np.random.choice(["basic", "ml"])
    env = VPNObfuscationEnv(dpi_strategy=dpi_strategy)
    return Monitor(env)

# --- Utility: Evaluate a model ---
def evaluate(model, env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards)

# --- Meta-Training Loop ---
meta_policy_path = "meta_policy_tmp"
for meta_iter in range(meta_iterations):
    meta_rewards = []
    # Save meta-policy state
    if os.path.exists(meta_policy_path):
        shutil.rmtree(meta_policy_path)
    os.makedirs(meta_policy_path, exist_ok=True)

    # Create and save a fresh meta-policy
    meta_env = make_random_env()
    meta_model = SAC(
        policy="MlpPolicy",
        env=meta_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=0,
        policy_kwargs=policy_kwargs
    )
    meta_model.save(f"{meta_policy_path}/meta_model")

    for task in range(meta_batch_size):
        # --- Clone meta-policy for this task ---
        task_env = make_random_env()
        task_model = SAC.load(f"{meta_policy_path}/meta_model", env=task_env)

        # --- Inner loop: Adaptation ---
        task_model.learn(total_timesteps=adapt_steps_timesteps, reset_num_timesteps=False, progress_bar=False)

        # --- Outer loop: Evaluate adapted agent ---
        avg_reward = evaluate(task_model, task_env, n_episodes=eval_episodes)
        meta_rewards.append(avg_reward)

    # --- Meta-update: Average reward (for logging) ---
    print(f"Meta-iteration {meta_iter}: Avg post-adaptation reward across tasks: {np.mean(meta_rewards):.2f}")

    # (True meta-gradient update is not supported in SB3; you can try first-order MAML by averaging weights, but this is a research topic.)

print("Meta-training done.")

# --- Test adaptation on a new, unseen task ---
test_env = make_random_env()
test_model = SAC.load(f"{meta_policy_path}/meta_model", env=test_env)
before = evaluate(test_model, test_env, n_episodes=eval_episodes)
print(f"Test reward before adaptation: {before:.2f}")

# Adaptation step(s)
test_model.learn(total_timesteps=adapt_steps_timesteps, reset_num_timesteps=False, progress_bar=False)
after = evaluate(test_model, test_env, n_episodes=eval_episodes)
print(f"Test reward after adaptation: {after:.2f}")