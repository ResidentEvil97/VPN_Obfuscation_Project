"""
Clean and Merge CSV Data

This script takes the raw VPN and non-VPN CSVs, combines and shuffles them,
selects the features we care about, removes any rows with missing or invalid
values, and saves the cleaned sample to a new CSV file.

The script assumes that the raw CSVs are in the "data/processed" directory.
"""

import pandas as pd
import numpy as np
import os

# Load raw VPN and non-VPN CSVs
vpn = pd.read_csv("data/processed/vpn.csv")
nonvpn = pd.read_csv("data/processed/nonvpn.csv")

# Combine and shuffle
# Concatenate the two dataframes, shuffle the result, and reset the index
df = pd.concat([vpn, nonvpn]).sample(frac=1).reset_index(drop=True)

# Select only features we care about
# Select the columns we want to keep: flowBytesPerSecond, mean_flowiat, and label
df_clean = df[["flowBytesPerSecond", "mean_flowiat", "label"]]

# Remove any rows with missing or invalid values
# Replace any infinite or NaN values with pd.NA, and then drop any rows with pd.NA
df_clean = df_clean.replace([np.inf, -np.inf], pd.NA).dropna()

# Save cleaned sample
# Save the cleaned dataframe to a new CSV file
os.makedirs("data/processed", exist_ok=True)
df_clean.to_csv("data/processed/sample_balanced.csv", index=False)

print("Saved cleaned dataset with shape:", df_clean.shape)
