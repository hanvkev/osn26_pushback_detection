#!/usr/bin/env python3
"""Export the manually labelled test set as a self-contained Zenodo dataset.

Reads the fetched trajectories (``data/test/01_fetched``) and the labeller output
(``data/test/02_labeled``) and writes a flat, tool-independent release without
any OpenSky data. Users rebuild the trajectories with ``fetch_trajectories.py``:

    <name>/
    ├── README.md
    ├── fetch_trajectories.py
    ├── labels/flights.csv            one row per flight, incl. fetch window and fingerprints
    ├── labels/segments.csv           one row per labelled segment
    ├── labels/query_bounds.csv       bounding box per airport-day for the OpenSky query
    └── checksums.sha256

plus ``<name>.zip`` next to the folder. Segment indices are validated against
their timestamps, and free-text tags are mapped to a fixed vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from fetch_trajectories import SEGMENT_TYPES, fingerprint, pushback_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = PROJECT_ROOT / "data" / "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "zenodo"
DEFAULT_NAME = "pushback-adsb-testset-v1.0"
BOUNDS_MARGIN_DEG = 1e-6  # ~0.1 m, so that boundary points are inside however the query compares

# Labeller tags (free text, grown during labelling) -> released vocabulary.
TAG_MAP = {
    "late_transponder_activation": "late_transponder_activation",
    "missing_data": "missing_data",
    "noisy": "noisy",
    "PB End Noisy": "noisy_pushback_end",
    "Noise before PB start": "noise_before_pushback",
    "Noise before PB Start": "noise_before_pushback",
    "Only Pull": "only_pull",
    "Arrival": "arrival",
    "Difficult": "difficult",
    "Multiple PB": "multiple_pushbacks",
    "Multiple PB in short time": "multiple_pushbacks",
    "Could be just Noise": "uncertain_pushback",
    "Maybe Short PB": "uncertain_pushback",
    "Maybe Short PB or just Noise": "uncertain_pushback",
    "Contains resampled segments": "resampled_within_pushback",
    "Resampled segment before PB start": "resampled_before_pushback",
    "Resampled Segment Before PB Start": "resampled_before_pushback",
    "Starts with a resampled segment": "starts_with_resampled_segment",
    "Resampled Segment after PB": "resampled_after_pushback",
    "Resampled Segment After PB End": "resampled_after_pushback",
    "Resampled Segment End (Excluded)": "resampled_end_excluded",
    "Resampled PB End (Excluded)": "resampled_end_excluded",
    "Resampled Segment End (Included)": "resampled_end_included",
    "Resampled PB End (Included)": "resampled_end_included",
    # used before the included/excluded distinction existed; mostly an artefact after the pushback
    "Resampled segment end": "resampled_after_pushback",
    "Complete PB resampled": "resampled_complete",
    "Complete PB Resampled": "resampled_complete",
    "PB Consists Only of a Resampled Segment": "resampled_complete",
    "Resampling drift at the start": "resampling_drift",
    "Resample Drift out of Position (Included)": "resampling_drift",
}


NORTHERN_SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                    6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
OPPOSITE_SEASON = {"Winter": "Summer", "Summer": "Winter", "Spring": "Autumn", "Autumn": "Spring"}


def local_season(date: pd.Timestamp, latitude: float) -> str:
    """Meteorological season at the airport (swapped on the southern hemisphere)."""
    season = NORTHERN_SEASONS[date.month]
    return OPPOSITE_SEASON[season] if latitude < 0 else season


def labelled_days() -> list[tuple[str, str, str, pd.Timestamp]]:
    """Return (airport, season, folder date, ISO date) for every labelled day."""
    days = []
    for path in sorted(TEST_ROOT.glob("02_labeled/*/*/*/annotations.json")):
        airport, season, day = path.parts[-4:-1]
        days.append((airport, season, day, pd.to_datetime(day, format="%d-%m-%y")))
    return days


# Per-flight exceptions to TAG_MAP: flight_id -> {labeller tag: released tag}.
FLIGHT_TAG_OVERRIDES: dict[str, dict[str, str]] = {}

TAG_DESCRIPTIONS = {
    "resampled_after_pushback": "The segment following the pushback is a resampling artefact (straight "
                                "interpolated line); the pushback end is not affected.",
    "resampled_end_excluded": "A resampling artefact starts during the pushback and hides its end; the "
                              "pushback end was set at the start of the artefact.",
    "resampled_end_included": "A short resampling artefact at the end of the pushback, after which the end "
                              "was still visible; the artefact is kept inside the pushback.",
    "late_transponder_activation": "The trajectory starts after the pushback had already begun, e.g. because "
                                   "the transponder was switched on late.",
    "noise_before_pushback": "Position noise (apparent movement) while the aircraft is still parked before "
                             "the pushback.",
    "resampled_complete": "The whole pushback consists of a resampling artefact (straight interpolated line).",
    "resampled_before_pushback": "A resampling artefact occurs before the pushback start.",
    "resampled_within_pushback": "The pushback contains one or more resampling artefacts.",
    "only_pull": "Only a pull movement is observed, no push phase.",
    "noisy": "Noisy positions within the pushback.",
    "arrival": "The flight looks like an arrival rather than a departure.",
    "multiple_pushbacks": "More than one pushback-like manoeuvre was observed; only one is labelled.",
    "uncertain_pushback": "Possibly a short pushback or only position noise.",
    "missing_data": "Parts of the trajectory are missing.",
    "starts_with_resampled_segment": "The pushback starts with a resampling artefact.",
    "resampling_drift": "The interpolated trajectory drifts slowly out of the parking position before a "
                        "clear movement begins; the actual pushback start lies somewhere within this drift "
                        "and cannot be determined exactly.",
    "noisy_pushback_end": "Noisy positions at the end of the pushback.",
    "difficult": "Difficult case for labelling.",
}


def map_tags(tags: list[str], flight_id: str) -> list[str]:
    tag_map = {**TAG_MAP, **FLIGHT_TAG_OVERRIDES.get(flight_id, {})}
    unknown = [t for t in tags if t not in tag_map]
    if unknown:
        raise KeyError(f"{flight_id}: unmapped tags {unknown}; extend TAG_MAP")
    return sorted({tag_map[t] for t in tags})


def query_bounds(traj: pd.DataFrame) -> list[float]:
    """Bounding box (west, south, east, north) for re-fetching the raw data of one airport-day.

    The original query used the OSM airport outline, which changes over time. Resampling
    interpolates linearly, so the stored points span exactly the raw positions inside the
    original box: their extent selects the same raw rows, independent of OSM.
    """
    return [
        traj["longitude"].min() - BOUNDS_MARGIN_DEG, traj["latitude"].min() - BOUNDS_MARGIN_DEG,
        traj["longitude"].max() + BOUNDS_MARGIN_DEG, traj["latitude"].max() + BOUNDS_MARGIN_DEG,
    ]


def export_day(airport: str, season: str, day: str, date: pd.Timestamp):
    """Return (flight rows, segment rows, query bounds) of one labelled airport-day."""
    annotations = json.loads((TEST_ROOT / "02_labeled" / airport / season / day / "annotations.json").read_text())
    traj = pd.read_parquet(TEST_ROOT / "01_fetched" / airport / season / f"{day}.parquet")

    # Same point order as the labeller: drop missing timestamps, sort per flight.
    traj = traj.dropna(subset=["timestamp"]).sort_values(["flight_id", "timestamp"], kind="stable")
    traj["point_idx"] = traj.groupby("flight_id").cumcount()
    traj = traj.reset_index(drop=True)

    if set(traj["flight_id"]) != set(annotations):
        raise ValueError(f"{airport} {day}: flight IDs of trajectories and labels differ")

    # traffic's assign_id counts per day, so IDs repeat across days; prefix airport and date.
    prefix = f"{airport}_{date:%Y-%m-%d}_"
    traj["flight_id"] = prefix + traj["flight_id"]
    annotations = {prefix + fid: ann for fid, ann in annotations.items()}

    timestamps = traj.set_index(["flight_id", "point_idx"])["timestamp"]
    per_flight = traj.groupby("flight_id").agg(
        icao24=("icao24", "first"), callsign=("callsign", "first"),
        first_ts=("timestamp", "min"), last_ts=("timestamp", "max"), n_points=("point_idx", "size"),
    )
    points = dict(tuple(traj.groupby("flight_id")))

    # The data folders use the northern-hemisphere season for every airport.
    if NORTHERN_SEASONS[date.month] != season:
        raise ValueError(f"{airport} {day}: folder season {season} does not match the date")
    season = local_season(date, traj["latitude"].median())

    flights, segments = [], []
    for flight_id, ann in sorted(annotations.items()):
        flight_segments = []
        for kind in SEGMENT_TYPES:
            for no, seg in enumerate(sorted(ann["segments"][kind], key=lambda s: s["start_idx"]), start=1):
                start_ts, end_ts = pd.Timestamp(seg["start_ts"]), pd.Timestamp(seg["end_ts"])
                if (timestamps[flight_id, seg["start_idx"]] != start_ts
                        or timestamps[flight_id, seg["end_idx"]] != end_ts):
                    raise ValueError(f"{flight_id}: {kind} segment {no} index/timestamp mismatch")
                flight_segments.append({
                    "flight_id": flight_id, "segment_type": kind, "segment_no": no,
                    "start_ts": start_ts, "end_ts": end_ts,
                    "start_idx": seg["start_idx"], "end_idx": seg["end_idx"],
                    "duration_s": (end_ts - start_ts).total_seconds(),
                })
        segments += flight_segments

        info = per_flight.loc[flight_id]
        flight_points = points[flight_id].reset_index(drop=True)
        flights.append({
            "flight_id": flight_id, "source_flight_id": flight_id.removeprefix(prefix), "airport": airport,
            "date": f"{date:%Y-%m-%d}", "season": season, "icao24": info.icao24, "callsign": info.callsign,
            "first_ts": info.first_ts, "last_ts": info.last_ts, "n_points": info.n_points,
            "pos_sha256": fingerprint(flight_points),
            "pb_pos_sha256": pushback_fingerprint(
                flight_points, pd.DataFrame(flight_segments, columns=["segment_type", "start_idx", "end_idx"])),
            "status": ann["status"],
            **{f"n_{kind}": len(ann["segments"][kind]) for kind in SEGMENT_TYPES},
            "tags": ";".join(map_tags(ann["tags"], flight_id)),
        })
    west, south, east, north = query_bounds(traj)
    bounds = {"airport": airport, "date": f"{date:%Y-%m-%d}", "west": west, "south": south, "east": east, "north": north}
    return flights, segments, bounds


def statistics_table(flights: pd.DataFrame) -> str:
    stats = flights.groupby(["airport", "season", "date"]).agg(
        flights=("flight_id", "size"),
        labelled=("status", lambda s: (s == "labelled").sum()),
        no_pushback=("status", lambda s: (s == "no_pushback").sum()),
        not_clear=("status", lambda s: (s == "not_clear").sum()),
        with_pull=("n_pull", lambda s: (s > 0).sum()),
    ).reset_index()
    total = stats.sum(numeric_only=True)
    lines = ["| Airport | Season | Date | Flights | labelled | no_pushback | not_clear | with pull |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.airport} | {r.season} | {r.date} | {r.flights} | {r.labelled} | {r.no_pushback} "
              f"| {r.not_clear} | {r.with_pull} |" for r in stats.itertuples()]
    lines.append(f"| **Total** | | | **{total.flights}** | **{total.labelled}** | **{total.no_pushback}** "
                 f"| **{total.not_clear}** | **{total.with_pull}** |")
    return "\n".join(lines)


def tag_table(flights: pd.DataFrame) -> str:
    counts = flights["tags"].str.split(";").explode().loc[lambda s: s != ""].value_counts()
    lines = ["| Tag | Flights | Description |", "|---|---:|---|"]
    lines += [f"| `{tag}` | {n} | {TAG_DESCRIPTIONS.get(tag, 'TODO')} |" for tag, n in counts.items()]
    return "\n".join(lines)


def write_readme(out: Path, flights: pd.DataFrame, segments: pd.DataFrame) -> None:
    template = (Path(__file__).with_name("README_template.md")).read_text()
    readme = template.format(
        n_flights=len(flights), n_segments=len(segments), n_days=flights["date"].nunique(),
        statistics_table=statistics_table(flights), tag_table=tag_table(flights),
    )
    (out / "README.md").write_text(readme)


def write_checksums(out: Path) -> None:
    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out).as_posix()}")
    (out / "checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    args = parser.parse_args()

    out = args.output / args.name
    if out.exists():
        shutil.rmtree(out)
    (out / "labels").mkdir(parents=True)

    flights, segments, bounds = [], [], []
    for airport, season, day, date in labelled_days():
        print(f"Exporting {airport} {date:%Y-%m-%d}")
        day_flights, day_segments, day_bounds = export_day(airport, season, day, date)
        flights += day_flights
        segments += day_segments
        bounds.append(day_bounds)

    flights = pd.DataFrame(flights)
    segments = pd.DataFrame(segments)
    flights.to_csv(out / "labels" / "flights.csv", index=False)
    segments.to_csv(out / "labels" / "segments.csv", index=False)
    pd.DataFrame(bounds).to_csv(out / "labels" / "query_bounds.csv", index=False, float_format="%.7f")

    shutil.copy(Path(__file__).with_name("fetch_trajectories.py"), out / "fetch_trajectories.py")
    write_readme(out, flights, segments)
    write_checksums(out)

    archive = shutil.make_archive(str(out), "zip", root_dir=args.output, base_dir=args.name)
    print(f"{len(flights)} flights, {len(segments)} segments -> {out}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
