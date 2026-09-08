"""Erzeugt einen kleinen CSV-Datensatz zum Testen der App."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(7)
    frames = []
    for flight_number in range(1, 7):
        count = 480
        timestamp = pd.date_range(
            f"2026-04-{flight_number:02d} 08:00:00",
            periods=count,
            freq="s",
            tz="Europe/Zurich",
        )
        direction = -1 if flight_number % 2 else 1
        movement = np.r_[np.zeros(35), np.linspace(0, direction * 0.0012, 150), np.full(count - 185, direction * 0.0012)]
        latitude = 47.4580 + movement + rng.normal(0, 0.000003, count)
        longitude = 8.5550 + movement * 0.45 + rng.normal(0, 0.000003, count)
        compute_gs = np.r_[np.zeros(35), np.abs(np.sin(np.linspace(0, 2 * np.pi, 150))) * 4, np.full(count - 185, 7.5)]
        frames.append(pd.DataFrame({
            "flight_id": f"DEMO_{flight_number:03d}",
            "timestamp": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "compute_gs": compute_gs,
            "compute_track": np.mod(180 + direction * np.linspace(0, 100, count), 360),
            "airport": "LSZH",
            "season": "spring",
        }))
    output = Path(__file__).with_name("demo_flights.csv")
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
