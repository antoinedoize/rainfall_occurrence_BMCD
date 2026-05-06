"""Generate synthetic toy precipitation CSVs for pipeline testing.

Produces 5 station CSVs in toy_csv_data/ using a seasonal Markov chain that
mimics Mediterranean rainfall patterns (dry summers, wet winters).

Output columns:
  date   - YYYYMMDD integer
  precip - daily precipitation in 0.1 mm units (integer)
           ≤ 6 = dry day, > 6 = wet day  (WET_DAY_THRESHOLD = 6)
"""
import pathlib
import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path(__file__).parent / "toy_csv_data"
OUT_DIR.mkdir(exist_ok=True)

STATIONS = {
    "ATHENS":     0,
    "BARCELONA":  1,
    "LISBON":     2,
    "MARSEILLE":  3,
    "ROME":       4,
}

START_DATE = "1950-01-01"
END_DATE   = "2020-12-31"

# Seasonal dry->wet transition probabilities (spring, summer, autumn, winter)
P_DRY_TO_WET = {3: 0.15, 4: 0.15, 5: 0.15,   # spring
                6: 0.05, 7: 0.04, 8: 0.05,    # summer
                9: 0.18, 10: 0.22, 11: 0.22,  # autumn
                12: 0.22, 1: 0.22, 2: 0.20}   # winter

P_WET_TO_DRY = 0.50  # constant: wet spells average ~2 days

GAMMA_SHAPE = 1.5
GAMMA_SCALE = 15.0  # mean ≈ 22.5 units = 2.25 mm; most values 7-60


def generate_station(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n = len(dates)
    months = dates.month.to_numpy()

    # Markov chain for wet/dry state
    state = np.zeros(n, dtype=int)  # 0 = dry, 1 = wet
    state[0] = rng.integers(0, 2)
    for i in range(1, n):
        m = months[i]
        if state[i - 1] == 0:
            state[i] = int(rng.random() < P_DRY_TO_WET[m])
        else:
            state[i] = int(rng.random() > P_WET_TO_DRY)

    # Precipitation amounts on wet days (min 7 so they clear WET_DAY_THRESHOLD=6)
    raw = rng.gamma(GAMMA_SHAPE, GAMMA_SCALE, size=n)
    precip = np.where(state == 1, np.maximum(7, raw.astype(int)), 0)

    return pd.DataFrame({
        "date":   dates.strftime("%Y%m%d").astype(int),
        "precip": precip.astype(int),
    })


def main():
    for station, idx in STATIONS.items():
        df = generate_station(seed=42 + idx * 17)
        out = OUT_DIR / f"{station}.csv"
        df.to_csv(out, index=False)
        wet = (df["precip"] > 6).sum()
        dry = (df["precip"] <= 6).sum()
        print(f"{station:12s}  rows={len(df):6d}  wet={wet:5d} ({100*wet/len(df):.1f}%)  "
              f"dry={dry:5d} ({100*dry/len(df):.1f}%)  -> {out.name}")
    print(f"\nDone. {len(STATIONS)} CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
