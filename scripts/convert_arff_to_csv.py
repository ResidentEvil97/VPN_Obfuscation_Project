"""
Convert ARFF files to CSV files for VPN/Non-VPN dataset

This script converts the raw ARFF files from the TimeBasedFeatures-Dataset-120s dataset
into CSV files that can be easily read by Pandas. The `convert_arff_to_csv` function
takes the path to an ARFF file, a label (0 for Non-VPN and 1 for VPN), and the output
CSV path as arguments. The function uses the `arff` library to load the ARFF file,
creates a Pandas DataFrame from the data, adds a "label" column with the given label,
and then saves the DataFrame to the output CSV path.

The script then uses this function to convert the raw VPN and Non-VPN ARFF files to
CSV files.
"""

import os
from scipy.io import arff
import pandas as pd

def convert_arff_to_csv(arff_path, label, output_csv):
    """
    Convert an ARFF file to a CSV file.

    Args:
        arff_path (str): The path to the ARFF file to convert.
        label (int): The label to assign to the output CSV file (0 for Non-VPN, 1 for VPN).
        output_csv (str): The path to save the output CSV file to.

    Returns:
        None
    """
    # Load the ARFF file using the arff library
    data, meta = arff.loadarff(arff_path)
    # Create a Pandas DataFrame from the data
    df = pd.DataFrame(data)
    # Add a "label" column to the DataFrame with the given label
    df["label"] = label
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save the DataFrame to the output CSV path
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

# The paths to the raw ARFF files
vpn_arff = "data/raw/TimeBasedFeatures-Dataset-120s-VPN.arff"
nonvpn_arff = "data/raw/TimeBasedFeatures-Dataset-120s-NO-VPN.arff"

# Convert the raw ARFF files to CSV files
convert_arff_to_csv(vpn_arff, 1, "data/processed/vpn.csv")
convert_arff_to_csv(nonvpn_arff, 0, "data/processed/nonvpn.csv")
