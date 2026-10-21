#!/usr/bin/env python3
"""Run the paper evaluation on the Zenodo release of the test set.

The evaluation logic is not duplicated: the code cells of ``paper_evaluation.ipynb``
are executed unchanged. Only the dataset manifest is replaced, so that

- trajectories are read from ``<dataset>/trajectories/<ICAO>_<YYYY-MM-DD>.parquet``
  (created by ``fetch_trajectories.py`` of the release), and
- ground truth is read from ``<dataset>/labels/``, converted per airport-day into the
  ``annotations.csv`` format of the labeller and parsed by the notebook's own
  ``load_ground_truth``.

Dataset IDs are the same as in the original evaluation (``<ICAO>/<Season>/<DD-MM-YY>``),
so the aggregated paper tables can be compared directly with ``--compare-with``.

Usage:
    python paper_evaluation_zenodo.py --dataset data/zenodo/pushback-adsb-testset-v1.0 \
        --compare-with results/evaluation/paper_results/without_not_clear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).with_name("paper_evaluation.ipynb")
MANIFEST_CALL = "manifest = discover_dataset_manifest()\n"
SEGMENT_TYPES = ["pushback", "push", "pull"]


def write_ground_truth(dataset: Path, output: Path) -> pd.DataFrame:
    """Write one labeller-style annotations.csv per airport-day and return the manifest."""
    flights = pd.read_csv(dataset / "labels" / "flights.csv", parse_dates=["first_ts", "last_ts"])
    segments = pd.read_csv(dataset / "labels" / "segments.csv", parse_dates=["start_ts", "end_ts"])
    segments_by_flight = dict(tuple(segments.groupby("flight_id")))

    rows = []
    for (airport, season, date), day in flights.groupby(["airport", "season", "date"]):
        day_name = f"{pd.Timestamp(date):%d-%m-%y}"
        dataset_id = f"{airport}/{season}/{day_name}"
        trajectory_path = dataset / "trajectories" / f"{airport}_{date}.parquet"
        ground_truth_path = output / "ground_truth" / airport / season / day_name / "annotations.csv"

        annotations = []
        for flight in day.itertuples():
            flight_segments = segments_by_flight.get(flight.flight_id, segments.iloc[:0])
            annotation = {"flight_id": flight.flight_id, "status": flight.status}
            for kind in SEGMENT_TYPES:
                annotation[f"{kind}_segments"] = json.dumps([
                    {"start_idx": int(s.start_idx), "end_idx": int(s.end_idx),
                     "start_ts": s.start_ts.isoformat(), "end_ts": s.end_ts.isoformat()}
                    for s in flight_segments[flight_segments["segment_type"] == kind].itertuples()
                ])
            annotation["tags"] = json.dumps([] if pd.isna(flight.tags) else flight.tags.split(";"))
            annotations.append(annotation)

        ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(annotations).to_csv(ground_truth_path, index=False)
        rows.append({
            "dataset_id": dataset_id, "airport": airport, "season": season, "day": day_name,
            "trajectory_path": str(trajectory_path), "ground_truth_path": str(ground_truth_path),
            "status": "ready" if trajectory_path.exists() else "missing_trajectories",
        })
    return pd.DataFrame(rows)


def run_notebook(dataset: Path, output: Path, workers: int | None = None) -> None:
    cells = [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    # Execute in the real __main__ namespace: the notebook passes its functions to worker
    # processes, which pickle them by reference to __main__.
    namespace = sys.modules["__main__"].__dict__
    namespace["display"] = print

    # Outside a notebook, plotly's fig.show() starts a local server and blocks until a
    # browser fetches the figure; the figures are only displayed, never saved.
    import plotly.basedatatypes
    plotly.basedatatypes.BaseFigure.show = lambda self, *args, **kwargs: None

    # Cell 1 sets the paths; redirect all outputs so the original results stay untouched.
    exec(cells[0], namespace)
    namespace["OUTPUT_ROOT"] = output
    namespace["CACHE_ROOT"] = output / "cache"
    namespace["UNIT_OUTPUT_ROOT"] = output / "units"
    if workers:
        namespace["PUSHBACK_WORKERS"] = workers

    # Cell 3 defines and immediately calls discover_dataset_manifest(); swap in the Zenodo manifest.
    if cells[1].count(MANIFEST_CALL) != 1:
        raise RuntimeError("paper_evaluation.ipynb changed: manifest call not found exactly once")
    definitions, remainder = cells[1].split(MANIFEST_CALL)
    exec(definitions, namespace)
    zenodo_manifest = write_ground_truth(dataset, output)
    sort_key = namespace["season_sort_key"]
    namespace["discover_dataset_manifest"] = lambda: zenodo_manifest.sort_values(
        ["airport", "season", "day", "dataset_id"],
        key=lambda column: column.map(sort_key) if column.name == "season" else column,
    ).reset_index(drop=True)
    exec(MANIFEST_CALL + remainder, namespace)

    for cell in cells[2:]:
        exec(cell, namespace)


def compare(new_dir: Path, reference_dir: Path) -> bool:
    """Compare the aggregated paper tables (CSV) of two evaluation runs."""
    identical = True
    for reference in sorted(reference_dir.glob("*.csv")):
        new = new_dir / reference.name
        if not new.exists():
            print(f"MISSING  {reference.name}")
            identical = False
            continue
        a, b = pd.read_csv(reference), pd.read_csv(new)
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)
            print(f"EQUAL    {reference.name}")
        except AssertionError as error:
            print(f"DIFFERS  {reference.name}\n{error}\n")
            identical = False
    return identical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Zenodo release folder with labels/ and trajectories/")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "evaluation_zenodo")
    parser.add_argument("--compare-with", type=Path, help="paper_results/<variant> folder of the original evaluation")
    parser.add_argument("--workers", type=int, help="flight-wise workers (notebook default: min(CPUs, 20))")
    args = parser.parse_args()

    run_notebook(args.dataset.resolve(), args.output.resolve(), args.workers)

    if args.compare_with:
        variant = args.compare_with.name
        identical = compare(args.output / "paper_results" / variant, args.compare_with)
        print("\nResults identical to the original evaluation." if identical
              else "\nResults differ from the original evaluation.")


if __name__ == "__main__":
    main()
