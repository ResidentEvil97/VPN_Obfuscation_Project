# VPN Obfuscation Project

**Overview**
This project implements a custom reinforcement learning environment for VPN traffic obfuscation, designed to evade a variety of Deep Packet Inspection (DPI) strategies. It trains a Soft Actor-Critic (SAC) agent to learn how to modify packet timing (jitter) and packet size in order to minimize detection. The environment simulates both rule-based and machine learning-based DPI adversaries, allowing for robust evaluation of the agent’s adaptability. The framework also supports meta-learning via Model-Agnostic Meta-Learning (MAML) for research into rapid adaptation to evolving censorship strategies.

**Project Goals**

```{verbatim}
Simulate rule-based and noisy DPI environments
Train a reinforcement learning agent using SAC
Add meta-learning with MAML for adaptability to evolving censorship strategies
Evaluate performance across different DPI heuristics
Visualize detection rates and average rewards across environments
```

**Repository Structure**

```{verbatim}
vpn/
├── rl/                                     # Reinforcement learning logic and environment
│   ├── vpn_env.py                          # Custom Gymnasium environment for VPN obfuscation
│   ├── sac_maml_trainer.py                 # SAC training script
│   ├── sac_maml_vpn.py                     # MAML meta-learning script for adaptability
│
├── dpi/                                    # DPI logic and simulated detection models
│   ├── mock_dpi.py                         # Heuristic detection rules for DPI
│   ├── packet_features.py                  # Feature extraction from PCAP or CSV
│   ├── tshark_dpi.py                       # Real-time DPI simulation using TShark (future/optional)
│   ├── data/                               # Data for DPI (used by packet_features or ML-DPI)
│   │   └── sample_combined_balanced.csv    # Example CSV dataset for DPI/ML classifier
│   └── models/                             # Machine learning classifiers for DPI logic (e.g., .pkl files)
│       └── (DPI classifier/model files)
│
├── scripts/                                # Run and evaluate experiments
│   ├── evaluate_policy.py                  # Evaluate trained agent across DPI strategies
│   └── run_experiment.py                   # (Optional) Experiment orchestration script
│
├── models/                                 # RL agent checkpoints (saved by SB3 or your RL scripts)
│   └── (RL agent .zip files)
│
├── logs/                                   # TensorBoard logs (ignored in git)
│   └── (TensorBoard event files)
│
├── test_env.py                             # Simple script to test environment behavior with random agent
├── README.md                               # Project overview and documentation
└── requirements.txt                        # Python dependencies
```

**How to Run**

```{verbatim}
Install Dependencies: pip3 install -r requirements.txt
Train the SAC agent: PYTHONPATH=. python3 rl/sac_maml_trainer.py
Evaluate Performance: PYTHONPATH=. python3 scripts/evaluate_policy.py
```

**Model Inputs and Actions**

```{verbatim}
State Representation:                  # What the agent observes at each step
    Mean packet latency                # Average time between packets (proxy for network delay)
    Packet size variation              # Standard deviation of recent packet sizes (traffic variability)
    Recent detection flag              # Whether the last step was detected by DPI (1.0 if detected, else 0.0)
    Packet size entropy                # Entropy of recent packet sizes (captures variability/unpredictability)

Action Space:                          # What the agent can control
    Delay to introduce (jitter)        # Amount of artificial delay added to packets (in ms)
    Packet size adjustment             # Amount to increase or decrease packet size (in bytes)
```

**Evaluation Metrics**

```{verbatim}
Average reward per step
    # Measures the agent's overall performance, balancing evasion, latency, and penalties.
    # Higher values indicate better evasion and efficiency.
    # Example: An average reward of 2.0 means the agent is consistently evading detection and minimizing penalties.

DPI detection rate across strategies
    # The fraction of steps where the agent is detected by DPI (lower is better).
    # This is reported for each DPI strategy (e.g., "basic", "strict", "noisy", "ml").
    # Example:
    #   BASIC DPI — Detection Rate: 100%
    #   ML DPI    — Detection Rate: 1%
    # Indicates the agent is almost always detected by rule-based DPI, but rarely by ML-based DPI.

Adaptability under noisy or strict detection rules
    # Assesses how well the agent generalizes to new or more challenging DPI strategies.
    # Can be measured by evaluating the agent on unseen DPI types or with increased noise/strictness.
    # Example:
    #   After training on "ml", the agent is evaluated on "strict" and "noisy":
    #   STRICT DPI — Avg Reward: 0.8, Detection Rate: 100%
    #   NOISY DPI  — Avg Reward: 0.8, Detection Rate: 100%
    #   ML DPI     — Avg Reward: 2.0, Detection Rate: 1%
    # Shows strong performance against ML DPI, but limited adaptability to strict/noisy rules.
```

**Future Work**

```{verbatim}
Replace heuristic DPI with ML-based classifier trained on CIC VPN-nonVPN dataset
Connect to Scapy and TShark for real-time packet generation and detection
Introduce dynamic adversarial DPI that adapts during training
```
