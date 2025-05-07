from scipy.io import arff
import pandas as pd
import os

def convert_arff_to_csv(arff_path, label, output_csv):
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    df["label"] = label
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

vpn_arff = "data/raw/TimeBasedFeatures-Dataset-120s-VPN.arff"
nonvpn_arff = "data/raw/TimeBasedFeatures-Dataset-120s-NO-VPN.arff"

convert_arff_to_csv(vpn_arff, 1, "data/processed/vpn.csv")
convert_arff_to_csv(nonvpn_arff, 0, "data/processed/nonvpn.csv")
