# VPN Obfuscation Project
**Overview**
This project uses reinforcement learning to dynamically obfuscate VPN traffic in order to evade modern Deep Packet Inspection (DPI) systems. The system trains a Soft Actor-Critic (SAC) agent, wrapped with Model-Agnostic Meta-Learning (MAML), to learn how to modify packet timing and size to avoid detection. The project simulates DPI strategies and evaluates the agent’s ability to adapt and remain undetected.

**Project Goals**
Simulate rule-based and noisy DPI environments
Train a reinforcement learning agent using SAC
Add meta-learning with MAML for adaptability to evolving censorship strategies
Evaluate performance across different DPI heuristics
Visualize detection rates and average rewards across environments

**Repository Structure**

vpn/
├── rl/                     # Reinforcement learning logic
│   ├── vpn_env.py          # Custom Gymnasium environment
│   └── sac_maml_trainer.py # SAC + MAML training script
│
├── dpi/                    # DPI logic and simulated detection models
│   ├── mock_dpi.py         # Placeholder for heuristic detection rules
│   ├── packet_features.py  # Feature extraction from PCAP or CSV
│   └── tshark_dpi.py       # Real-time DPI simulation using TShark (future)
│
├── scripts/                # Run and evaluate experiments
│   ├── evaluate_policy.py  # Runs trained models against DPI variants
│   └── run_experiment.py   # Wrapper script (optional)
│
├── data/                   # Placeholder for input PCAPs or metadata
├── models/                 # Trained model checkpoints
├── logs/                   # TensorBoard logs (add to .gitignore)
├── test_env.py             # Simple test for verifying environment behavior
├── README.md               # Project overview and documentation
└── requirements.txt        # Python dependencies

**How to Run**
Install Dependencies: pip3 install -r requirements.txt
Train the SAC agent: PYTHONPATH=. python3 rl/sac_maml_trainer.py
Evaluate Performance: PYTHONPATH=. python3 scripts/evaluate_policy.py


**Model Inputs and Actions**
State Representation:
    Mean packet latency
    Packet size variation
    Recent detection flag
Action Space:
    Delay to introduce (jitter)
    Packet size adjustment

**Evaluation Metrics**
    Average reward per step
    DPI detection rate across strategies
    Adaptability under noisy or strict detection rules
    
**Future Work**
Replace heuristic DPI with ML-based classifier trained on CIC VPN-nonVPN dataset
Connect to Scapy and TShark for real-time packet generation and detection
Introduce dynamic adversarial DPI that adapts during training
