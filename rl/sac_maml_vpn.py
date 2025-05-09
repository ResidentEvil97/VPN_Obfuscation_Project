""" 
 =========================================================================== 
  Meta-Learning for VPN Traffic Obfuscation
 =========================================================================== 

  This script trains a Soft Actor-Critic (SAC) agent using Model-Agnostic
  Meta-Learning (MAML) to learn how to adapt to new VPN traffic obfuscation
  environments. The script is organized into a few sections:

  1. Hyperparameters: Meta-batch size, number of meta-iterations, inner
     adaptation steps, and evaluation episodes.
  2. Environment creation: A function to create a randomized VPN traffic
     obfuscation environment (task).
  3. Model evaluation: A function to evaluate a model in a given environment.
  4. Meta-training loop: The main loop that creates tasks, adapts the model,
     evaluates the adapted model, and logs the results.
  5. Test adaptation: Test the adapted model on a new, unseen task.
"""


# Import necessary libraries
# Numpy is used for numerical computations
import numpy as np
import os
import time
import shutil

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl.vpn_env import VPNObfuscationEnv



# Hyperparameters
meta_batch_size = 1                     # Number of tasks per meta-iteration
meta_iterations = 1                     # Number of meta-iterations
inner_steps = 1                         # Number of adaptation steps per task
adapt_steps_timesteps = 10              # Timesteps for each adaptation step
eval_episodes = 1                       # Evaluation episodes per task
policy_kwargs = dict(net_arch=[32])     # Policy architecture

# Create a randomized environment (TASK)
def make_random_env():
    """Create a randomized VPN traffic obfuscation environment (task).

    Returns:
        VPNObfuscationEnv: A randomized VPNObfuscationEnv instance.
    """
    dpi_strategy = "basic"  # Only use fast, non-ML DPI for debugging
    env = VPNObfuscationEnv(dpi_strategy=dpi_strategy)
    return Monitor(env)

# Evaluate a model
def evaluate(model, env, n_episodes=5):
    """Evaluate a model in a given environment.

    Args:
        model (SAC): The model to evaluate.
        env (VPNObfuscationEnv): The environment to evaluate in.
        n_episodes (int): The number of episodes to evaluate. Defaults to 5.

    Returns:
        float: The average reward across episodes.
    """
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < 100000:  # Prevent infinite loops
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
        if steps >= 100000:
            print("Warning: Evaluation episode hit max step limit (possible infinite loop)")
        rewards.append(ep_reward)
    return np.mean(rewards)

# Meta-training loop
meta_policy_path = "meta_policy_tmp"
for meta_iter in range(meta_iterations):
    print(f"Meta-iteration {meta_iter} starting")
    start = time.time()
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
        # Clone meta-policy for this task
        task_env = make_random_env()
        task_model = SAC.load(f"{meta_policy_path}/meta_model", env=task_env)

        # Inner loop: Adaptation
        start_adapt = time.time()
        task_model.learn(total_timesteps=adapt_steps_timesteps, reset_num_timesteps=False, progress_bar=False)
        end_adapt = time.time()
        print(f"Task {task} adaptation took {end_adapt - start_adapt:.2f} seconds")

        # Outer loop: Evaluate adapted agent
        start_eval = time.time()
        avg_reward = evaluate(task_model, task_env, n_episodes=eval_episodes)
        end_eval = time.time()
        print(f"Task {task} evaluation took {end_eval - start_eval:.2f} seconds")
        meta_rewards.append(avg_reward)

    # Meta-update: Average reward (for logging)
    print(f"Meta-iteration {meta_iter}: Avg post-adaptation reward across tasks: {np.mean(meta_rewards):.2f}")

# Test adaptation on a new, unseen task
test_env = make_random_env()
test_model = SAC.load(f"{meta_policy_path}/meta_model", env=test_env)
before = evaluate(test_model, test_env, n_episodes=eval_episodes)
print(f"Test reward before adaptation: {before:.2f}")

# Adaptation step(s)
test_model.learn(total_timesteps=adapt_steps_timesteps, reset_num_timesteps=False, progress_bar=False)
after = evaluate(test_model, test_env, n_episodes=eval_episodes)
print(f"Test reward after adaptation: {after:.2f}")