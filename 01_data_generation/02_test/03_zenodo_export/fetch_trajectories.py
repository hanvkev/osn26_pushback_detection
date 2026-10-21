#!/usr/bin/env python3
"""Fetch the trajectories of the pushback test set from OpenSky and attach the labels.

Requires access to the OpenSky historical database (https://opensky-network.org/data/trino),
configured for pyopensky. The trajectories are rebuilt exactly as for labelling:

    opensky.history(icao24, first_ts .. last_ts, bounding box of the airport-day)
    -> traffic resample("1s") -> cumulative_distance()

Every flight is checked against the position fingerprints in labels/flights.csv. The
result is stored per point (``match_flight``, ``match_pushback``) and in verification.csv.

Usage:
    python fetch_trajectories.py                      # all 16 airport-days
    python fetch_trajectories.py --airport KBOS --date 2025-10-18
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAJECTORY_COLUMNS = [
    "flight_id", "point_idx", "timestamp", "icao24", "callsign",
    "latitude", "longitude", "altitude", "geoaltitude", "vertical_rate",
    "groundspeed", "track", "compute_gs", "compute_track", "onground", "squawk",
]
SEGMENT_TYPES = ["pushback", "push", "pull"]


def fingerprint(traj: pd.DataFrame) -> str:
    """SHA-256 of the positions (``lat,lon`` with 6 decimals, one line per point)."""
    pos = traj["latitude"].map("{:.6f}".format) + "," + traj["longitude"].map("{:.6f}".format)
    return hashlib.sha256("\n".join(pos).encode()).hexdigest()


def pushback_fingerprint(traj: pd.DataFrame, segments: pd.DataFrame) -> str | None:
    """Fingerprint of the points from the first pushback start to the last pushback end."""
    pushback = segments[segments["segment_type"] == "pushback"]
    if pushback.empty:
        return None
    return fingerprint(traj.iloc[pushback["start_idx"].min():pushback["end_idx"].max() + 1])


def rebuild_flight(raw: pd.DataFrame, flight: pd.Series) -> pd.DataFrame:
    """Rebuild one flight from raw OpenSky state vectors (same steps as for labelling)."""
    from traffic.core import Flight

    # traffic's assign_id separates flights by icao24 and callsign, so rows without
    # the flight's callsign (e.g. missing callsign) are not part of the flight.
    first, last = pd.Timestamp(flight.first_ts), pd.Timestamp(flight.last_ts)
    data = raw[(raw["icao24"] == flight.icao24) & (raw["callsign"] == flight.callsign)
               & raw["timestamp"].between(first, last)].copy()
    data["flight_id"] = flight.flight_id
    traj = Flight(data).resample("1s").cumulative_distance().data
    traj = traj[traj["timestamp"].between(first, last)].sort_values("timestamp").reset_index(drop=True)
    traj["point_idx"] = np.arange(len(traj))
    return traj[TRAJECTORY_COLUMNS]


def fetch_raw(flights: pd.DataFrame, bbox: list[float]) -> pd.DataFrame:
    """Query all state vectors of one airport-day in one request."""
    from traffic.data import opensky

    result = opensky.history(
        start=pd.Timestamp(flights["first_ts"].min()),
        stop=pd.Timestamp(flights["last_ts"].max()) + pd.Timedelta("1s"),
        icao24=sorted(flights["icao24"].unique()),
        bounds=tuple(bbox),
    )
    if result is None:
        raise RuntimeError("OpenSky returned no data")
    raw = result.data.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    for col in ["onground", "alert", "spi"]:
        raw[col] = raw[col].astype(np.bool_)
    for col in ["latitude", "longitude", "groundspeed", "track", "vertical_rate",
                "altitude", "geoaltitude", "lastcontact", "last_position"]:
        raw[col] = raw[col].astype("double")
    for col in ["icao24", "callsign", "squawk"]:
        raw[col] = raw[col].astype("object")
    return raw


def add_labels(traj: pd.DataFrame, flight: pd.Series, segments: pd.DataFrame) -> pd.DataFrame:
    traj["status"] = flight.status
    for kind in SEGMENT_TYPES:
        traj[f"is_{kind}"] = False
        for seg in segments[segments["segment_type"] == kind].itertuples():
            traj.loc[seg.start_idx:seg.end_idx, f"is_{kind}"] = True
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=ROOT / "labels")
    parser.add_argument("--output", type=Path, default=ROOT / "trajectories")
    parser.add_argument("--airport")
    parser.add_argument("--date", help="YYYY-MM-DD")
    args = parser.parse_args()

    flights = pd.read_csv(args.labels / "flights.csv", parse_dates=["first_ts", "last_ts"])
    segments = pd.read_csv(args.labels / "segments.csv", parse_dates=["start_ts", "end_ts"])
    bounds = pd.read_csv(args.labels / "query_bounds.csv").set_index(["airport", "date"])
    if args.airport:
        flights = flights[flights["airport"] == args.airport]
    if args.date:
        flights = flights[flights["date"] == args.date]
    args.output.mkdir(parents=True, exist_ok=True)

    checks = []
    for (airport, date), day in flights.groupby(["airport", "date"]):
        print(f"{airport} {date}: fetching {len(day)} flights")
        raw = fetch_raw(day, bounds.loc[(airport, date), ["west", "south", "east", "north"]].tolist())
        trajs = []
        for flight in day.itertuples():
            traj = rebuild_flight(raw, flight)
            flight_segments = segments[segments["flight_id"] == flight.flight_id]
            # OpenSky occasionally returns a raw point at a gap edge slightly differently than at
            # labelling time, which changes the interpolation in that gap. Flag instead of drop.
            match_flight = len(traj) == flight.n_points and fingerprint(traj) == flight.pos_sha256
            match_pushback = (pd.NA if pd.isna(flight.pb_pos_sha256)
                              else len(traj) == flight.n_points
                              and pushback_fingerprint(traj, flight_segments) == flight.pb_pos_sha256)
            traj = add_labels(traj, flight, flight_segments)
            traj["match_flight"], traj["match_pushback"] = match_flight, match_pushback
            trajs.append(traj)
            checks.append((flight.flight_id, match_flight, match_pushback))
        day_checks = pd.DataFrame(checks[-len(day):], columns=["flight_id", "match_flight", "match_pushback"])
        pd.concat(trajs, ignore_index=True).to_parquet(args.output / f"{airport}_{date}.parquet", index=False)
        print(f"  identical: {day_checks['match_flight'].sum()}/{len(day)} flights, "
              f"{(day_checks['match_pushback'] == True).sum()}/{day_checks['match_pushback'].notna().sum()} pushbacks")  # noqa: E712

    checks = pd.DataFrame(checks, columns=["flight_id", "match_flight", "match_pushback"])
    checks.to_csv(args.output / "verification.csv", index=False)
    differing = checks[checks["match_pushback"] == False]  # noqa: E712
    if not differing.empty:
        print(f"\n{len(differing)} labelled pushbacks differ from the labelled data (see verification.csv):")
        print("\n".join(differing["flight_id"]))


if __name__ == "__main__":
    main()
