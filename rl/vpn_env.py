"""
VPN Obfuscation Environment
===========================

This file contains the implementation of the VPN obfuscation environment.

The environment is responsible for simulating the effects of a VPN on network traffic,
and for simulating the detection of the VPN by a DPI system.

The environment takes in an action, which is a list of two elements: the jitter to add
to the packet timing, and the modification to make to the packet size.

The environment then returns a tuple containing the new state, the reward, whether
the episode has terminated, and a dictionary of additional information.

The state of the environment is a tuple containing the mean latency, the standard
deviation of the packet size, a flag indicating whether the VPN was detected, and
the entropy of the packet size history.

The reward is a scalar value that is calculated based on the state of the environment.
The reward is higher if the VPN is not detected, and lower if the VPN is detected.

The environment also keeps track of the number of steps taken, and will terminate
the episode if the number of steps taken exceeds a certain maximum.

The environment is designed to be used with the Stable Baselines library, and
is intended to be used as a testbed for reinforcement learning algorithms.

"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
import joblib

from dpi.mock_dpi import MockDPI
class VPNObfuscationEnv(gym.Env):
    """
    Environment for the VPN Obfuscation task.

    The environment takes in an action, which is a list of two elements: the jitter
    to add to the packet timing, and the modification to make to the packet size.

    The environment then returns a tuple containing the new state, the reward,
    whether the episode has terminated, and a dictionary of additional information.
    """

    def __init__(self, dpi_strategy="basic"):
        """
        Initialize the environment.

        Args:
            dpi_strategy (str): The DPI strategy to use. Defaults to "basic".
        """
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
        """
        Reset the environment.

        Args:
            seed (int): The random seed to use. Defaults to None.
            options (dict): Additional options for the reset method. Defaults to None.

        Returns:
            tuple: The initial state and info.
        """
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
        """
        Take a step in the environment.

        Args:
            action (list): The action to take.

        Returns:
            tuple: The new state, reward, terminated, truncated, and info.
        """
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

        if self.dpi_strategy == "basic":
            if detected:
                reward -= 1.0  # less penalty for detection
            else:
                reward += 2.0  # Bigger bonus for evasion
        else:
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
        if self.dpi_strategy == "strict":
            return (jitter > 50 or size_mod > 30 or random.random() < 0.10)
        elif self.dpi_strategy == "noisy":
            return random.random() < 0.5
        elif self.dpi_strategy == "basic":
            return ((jitter + abs(size_mod)) > 130)
