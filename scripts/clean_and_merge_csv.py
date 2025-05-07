import pandas as pd
import numpy as np
import os

# Load raw VPN and non-VPN CSVs
vpn = pd.read_csv("data/processed/vpn.csv")
nonvpn = pd.read_csv("data/processed/nonvpn.csv")

# Combine and shuffle
df = pd.concat([vpn, nonvpn]).sample(frac=1).reset_index(drop=True)

# Select only features we care about
df_clean = df[["flowBytesPerSecond", "mean_flowiat", "label"]]

# Remove any rows with missing or invalid values
df_clean = df_clean.replace([np.inf, -np.inf], pd.NA).dropna()

# Save cleaned sample
os.makedirs("data/processed", exist_ok=True)
df_clean.to_csv("data/processed/sample_balanced.csv", index=False)

print("Saved cleaned dataset with shape:", df_clean.shape)
