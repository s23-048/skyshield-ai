# get_dataset.py
"""
Very simple version:
- Downloads FAA-like wildlife strike dataset (public mirror)
- Selects and renames columns
- DOES NOT drop rows (except if dataset truly empty)
- Saves: data/faa_bird_strike_sample_10k.csv

Output columns:
    AIRCRAFT
    PHASE_OF_FLIGHT
    SPECIES
    TIME_OF_DAY
    HEIGHT
    DAMAGE_SEVERITY
"""

import os
import pandas as pd

URL = "https://jsvine.github.io/intro-to-visidata/_downloads/a61d9b28e9a942e1254bffeb8289a447/faa-wildlife-strikes.csv"


def choose_col(df, options, name):
    for opt in options:
        if opt in df.columns:
            print(f"[INFO] Using column '{opt}' for {name}")
            return df[opt]
    print(f"[WARN] No column found for {name}, filling with 'UNKNOWN' or -1")
    return None


def main():
    print("Downloading FAA-style dataset from public mirror...")
    df = pd.read_csv(URL, low_memory=False)
    print(f"Original rows: {len(df)}")
    print("Original columns:", list(df.columns)[:20], "...")

    # Standardize columns to uppercase
    df.columns = [c.upper() for c in df.columns]

    # Pick source columns
    aircraft = choose_col(df, ["AIRCRAFT", "AIRCRAFT_MAKE_MODEL"], "AIRCRAFT")
    phase = choose_col(df, ["PHASE_OF_FLIGHT", "PHASE_OF_FLT", "PHASE"], "PHASE_OF_FLIGHT")
    species = choose_col(df, ["SPECIES", "BIRD_SPECIES"], "SPECIES")
    time_of_day = choose_col(df, ["TIME_OF_DAY", "TIME"], "TIME_OF_DAY")
    height = choose_col(df, ["HEIGHT", "ALTITUDE"], "HEIGHT")
    damage = choose_col(df, ["DAMAGE", "DAMAGE_LEVEL"], "DAMAGE")

    n = len(df)
    if n == 0:
        print("ERROR: Downloaded dataset has 0 rows. Exiting.")
        return

    # If any column is missing, fill with defaults
    if aircraft is None:
        aircraft = pd.Series(["UNKNOWN"] * n)
    if phase is None:
        phase = pd.Series(["UNKNOWN"] * n)
    if species is None:
        species = pd.Series(["UNKNOWN"] * n)
    if time_of_day is None:
        time_of_day = pd.Series(["UNKNOWN"] * n)
    if height is None:
        height = pd.Series([-1] * n)
    if damage is None:
        damage = pd.Series(["NONE"] * n)

    clean_df = pd.DataFrame(
        {
            "AIRCRAFT": aircraft.astype(str).str.strip(),
            "PHASE_OF_FLIGHT": phase.astype(str).str.strip(),
            "SPECIES": species.astype(str).str.strip(),
            "TIME_OF_DAY": time_of_day.astype(str).str.strip(),
            "HEIGHT": pd.to_numeric(height, errors="coerce"),
            "DAMAGE": damage.astype(str).str.strip(),
        }
    )

    # Fill missing height with -1
    clean_df["HEIGHT"] = clean_df["HEIGHT"].fillna(-1).astype(int)

    # Map DAMAGE -> DAMAGE_SEVERITY
    def map_damage(x):
        s = str(x).upper()
        if "MINOR" in s or s == "M":
            return 1
        if "SUBSTANTIAL" in s or "DESTROY" in s or s == "S":
            return 2
        return 0  # treat all else as none

    clean_df["DAMAGE_SEVERITY"] = clean_df["DAMAGE"].apply(map_damage)

    # Drop only the original DAMAGE text column
    clean_df = clean_df.drop(columns=["DAMAGE"])

    print(f"Rows after simple cleaning (no dropping): {len(clean_df)}")

    # Sample up to 10k rows (only if > 10k)
    if len(clean_df) > 10000:
        clean_df = clean_df.sample(10000, random_state=42)
        print("Sampled to 10,000 rows for faster training.")
    else:
        print("Dataset has <= 10,000 rows, using all of them.")

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "faa_bird_strike_sample_10k.csv")
    clean_df.to_csv(out_path, index=False)

    print("\nDataset prepared successfully!")
    print(f"Saved to: {out_path}")
    print(f"Final rows: {len(clean_df)}")
    print("Columns:", list(clean_df.columns))


if __name__ == "__main__":
    main()
