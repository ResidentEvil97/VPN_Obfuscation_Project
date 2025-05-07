import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dpi.mock_dpi import MockDPI
import pandas as pd
import joblib


class VPNObfuscationEnv(gym.Env):
    def __init__(self, dpi_strategy="basic"):
        super().__init__()

        # Define action space: [jitter_ms (0–200), packet_size_mod (-100 to +100 bytes)]
        self.action_space = spaces.Box(low=np.array([0.0, -100.0]),
                                       high=np.array([200.0, 100.0]),
                                       dtype=np.float32)

        # Define observation space: [mean_latency, std_packet_size, recent_detection]
        self.observation_space = spaces.Box(low=0.0,
                                            high=1000.0,
                                            shape=(3,),
                                            dtype=np.float32)

        # DPI detection strategy
        self.dpi_strategy = dpi_strategy
        # Load ML DPI model only if needed
        if dpi_strategy == "ml":
            self.dpi_model = joblib.load("dpi/models/random_forest_dpi.pkl")
        else:
            self.dpi = MockDPI()

        # Internal state
        self.state = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize starting traffic conditions
        mean_latency = np.random.uniform(40.0, 60.0)
        std_packet_size = np.random.uniform(150.0, 250.0)
        detection_flag = 0.0  # New episode → no detection yet

        self.state = np.array([mean_latency, std_packet_size, detection_flag], dtype=np.float32)
        return self.state, {}

    
    def step(self, action):
        # Unpack action: [jitter (ms), packet size modifier (bytes)]
        jitter, size_mod = action

        # Simulate DPI detection
        detected = self._simulate_dpi(jitter, size_mod)

        # Latency penalty
        latency_penalty = jitter / 100.0  # Scale down (0–2)

        # Reward logic
        if detected:
            reward = -1.0
        else:
            reward = 1.0

        reward -= latency_penalty  # penalize high delay even if undetected
        reward = float(reward)     # convert to native Python float ← ★ THIS FIX

        # Update internal state
        mean_latency = np.clip(self.state[0] + jitter, 0, 1000)
        std_packet_size = np.clip(self.state[1] + size_mod, 0, 1000)
        detection_flag = 1.0 if detected else 0.0

        self.state = np.array([mean_latency, std_packet_size, detection_flag], dtype=np.float32)

        terminated = False  # no terminal state for now
        truncated = False   # no episode cutoff

        return self.state, reward, terminated, truncated, {}


    
    def _simulate_dpi(self, jitter, size_mod):
        """
        Simulated DPI logic using rule-based or ML-based model.
        """
        if self.dpi_strategy == "ml":
            # Construct feature vector for the ML classifier
            duration = self.state[0]                     # mean_latency
            pkts_per_sec = jitter                        # proxy for transmission rate
            bytes_per_sec = self.state[1] + size_mod     # estimated flowBytesPerSecond
            features = pd.DataFrame([[duration, pkts_per_sec, bytes_per_sec]],
                        columns=["duration", "flowPktsPerSecond", "flowBytesPerSecond"])
            prediction = self.dpi_model.predict(features)[0]
            return bool(prediction)  # 1 = detected, 0 = not detected

        # Rule-based fallbacks
        elif self.dpi_strategy == "basic":
            return (jitter > 100 or abs(size_mod) > 50)
        elif self.dpi_strategy == "strict":
            return (jitter > 50 or size_mod > 30)
        elif self.dpi_strategy == "noisy":
            return np.random.rand() < 0.5
        else:
            return (jitter > 75 and abs(size_mod) > 75)




