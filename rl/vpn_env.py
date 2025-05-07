import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dpi.mock_dpi import MockDPI
import pandas as pd
import joblib
import random

class VPNObfuscationEnv(gym.Env):
    def __init__(self, dpi_strategy="basic"):
        super().__init__()

        # Define action space: [jitter_ms (0–200), packet_size_mod (-100 to +100 bytes)]
        self.action_space = spaces.Box(low=np.array([0.0, -100.0]),
                                       high=np.array([200.0, 100.0]),
                                       dtype=np.float32)

        # Define observation space: [mean_latency, std_packet_size, recent_detection]
        self.observation_space = spaces.Box(low=0.0, high=1000.0,
                                            shape=(4,), dtype=np.float32)

        # DPI detection strategy
        self.dpi_strategy = dpi_strategy
        # Load ML DPI model only if needed
        if dpi_strategy == "ml":
            self.dpi_model = joblib.load("dpi/models/random_forest_dpi.pkl")
        else:
            self.dpi = MockDPI()

        # Internal state
        self.history_length = 10
        self.packet_size_history = []
        self.state = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mean_latency = np.random.uniform(40.0, 60.0)
        std_packet_size = np.random.uniform(150.0, 250.0)
        detection_flag = 0.0
        packet_size_entropy = 0.0
        self.state = np.array([mean_latency, std_packet_size, detection_flag, packet_size_entropy], dtype=np.float32)
        self.packet_size_history = [std_packet_size]
        return self.state, {}

    
    def step(self, action):
        # --- Apply action to internal state ---
        jitter = float(action[0]) * 100  # Example scaling
        size_mod = float(action[1]) * 100

        # --- Simulate DPI detection ---
        detected = self._simulate_dpi(jitter, size_mod)

        # --- Update internal state ---
        mean_latency = np.clip(self.state[0] + jitter, 0, 1000)
        std_packet_size = np.clip(self.state[1] + size_mod, 0, 1000)
        self.packet_size_history.append(std_packet_size)
        if len(self.packet_size_history) > self.history_length:
            self.packet_size_history.pop(0)

        packet_size_entropy = float(np.std(self.packet_size_history))
        detection_flag = 1.0 if detected else 0.0

        self.state = np.array([mean_latency, std_packet_size, detection_flag, packet_size_entropy], dtype=np.float32)

        # --- Reward calculation ---
        # Encourage evasion, penalize detection, penalize high latency, small step penalty to encourage faster solutions
        reward = 5.0  # Increased base reward

        latency_penalty = mean_latency / 250.0    # Lower scaling, use updated latency
        step_penalty = 0.01                       # small negative reward per step

        if detected:
            reward -= 0.2  # Lower detection penalty
        else:
            reward += 1.0  # Bonus for evading detection

        reward -= latency_penalty
        reward -= step_penalty

        # --- Episode termination logic ---
        self.current_step += 1
        max_steps = 200
        terminated = False
        truncated = False

        # Optionally, terminate if detected (uncomment if desired):
        # if detected:
        #     terminated = True

        if self.current_step >= max_steps:
            truncated = True

        info = {
            "detected": detected,
            "latency_penalty": latency_penalty,
            "step_penalty": step_penalty
        }

        return self.state, reward, terminated, truncated, info
    
    def _simulate_dpi(self, jitter, size_mod):
        """
        Simulated DPI logic using rule-based or ML-based model.
        """
        if self.dpi_strategy == "ml":
            duration = self.state[0]                     # mean_latency
            pkts_per_sec = jitter                        # proxy for transmission rate
            bytes_per_sec = self.state[1] + size_mod     # estimated flowBytesPerSecond
            features = pd.DataFrame([[duration, pkts_per_sec, bytes_per_sec]],
                        columns=["duration", "flowPktsPerSecond", "flowBytesPerSecond"])
            prediction = self.dpi_model.predict(features)[0]
            return bool(prediction)  # 1 = detected, 0 = not detected

        # Rule-based fallbacks
        return (jitter > 90 or size_mod > 90 or random.random() < 0.05)

        # elif self.dpi_strategy == "basic":
        #     return (jitter > 100 or abs(size_mod) > 50)
        # elif self.dpi_strategy == "strict":
        #     return (jitter > 50 or size_mod > 30)
        # elif self.dpi_strategy == "noisy":
        #     return np.random.rand() < 0.5
        # else:
        #     return (jitter > 80 or size_mod > 80)

    def run_random_agent(num_episodes=5, max_steps=200):
        env = VPNObfuscationEnv()
        for ep in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            detected_steps = 0
            for t in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if info.get("detected"):
                    detected_steps += 1
                if terminated or truncated:
                    break
            print(f"RandomAgent Episode {ep+1}: Total reward = {total_reward:.2f}, Detected steps = {detected_steps}/{max_steps}")


