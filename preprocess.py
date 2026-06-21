# preprocess.py
import os
import pandas as pd


def load_bird_strike_data(csv_path: str = "data/faa_bird_strike_sample_10k.csv") -> pd.DataFrame:
    """
    Loads and lightly cleans the FAA bird strike dataset created by get_dataset.py.

    Expected columns in the CSV:
        AIRCRAFT
        PHASE_OF_FLIGHT
        SPECIES
        TIME_OF_DAY
        HEIGHT
        DAMAGE_SEVERITY  (0 = none, 1 = minor, 2 = severe)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run get_dataset.py first.")

    df = pd.read_csv(csv_path)

    # Standardize column names to uppercase for safety
    df.columns = [c.upper() for c in df.columns]

    required_cols = [
        "AIRCRAFT",
        "PHASE_OF_FLIGHT",
        "SPECIES",
        "TIME_OF_DAY",
        "HEIGHT",
        "DAMAGE_SEVERITY",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in dataset: {missing}. "
            "Check get_dataset.py or your CSV file."
        )

    # Basic cleaning
    for col in ["AIRCRAFT", "PHASE_OF_FLIGHT", "SPECIES", "TIME_OF_DAY"]:
        df[col] = df[col].astype(str).str.strip()

    # Ensure HEIGHT numeric
    df["HEIGHT"] = pd.to_numeric(df["HEIGHT"], errors="coerce").fillna(-1).astype(int)

    # Ensure DAMAGE_SEVERITY numeric int
    df["DAMAGE_SEVERITY"] = pd.to_numeric(df["DAMAGE_SEVERITY"], errors="coerce").fillna(0).astype(int)

    # Drop any rows with missing key categorical values
    df = df.dropna(subset=["AIRCRAFT", "PHASE_OF_FLIGHT", "SPECIES", "TIME_OF_DAY"])

    return df
