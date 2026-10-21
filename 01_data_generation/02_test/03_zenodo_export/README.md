# Zenodo Export

This directory builds the public release of the manually labelled test dataset:

> Hänggi, K. and Waltert, M. *Manually Labelled Aircraft Pushback Segments in ADS-B Trajectories*. Version 1.0. Zenodo. https://doi.org/10.5281/zenodo.23548217

The release contains only the labels. The trajectories are not redistributed; users rebuild them from the OpenSky historical database with `fetch_trajectories.py`, which reproduces the labelled points exactly and verifies them against position fingerprints.

## Files

- `export_zenodo_dataset.py` – builds the release from `data/test/01_fetched/` and `data/test/02_labeled/`
- `fetch_trajectories.py` – shipped with the release; fetches and rebuilds the trajectories and attaches point-wise labels
- `README_template.md` – template of the release `README.md`; statistics and tag table are filled in by the export
- `zenodo_description.md`, `zenodo_description.html` – metadata and description text for the Zenodo record

## Building the release

```bash
python export_zenodo_dataset.py
```

The release is written to:

```text
data/zenodo/pushback-adsb-testset-v1.0/
data/zenodo/pushback-adsb-testset-v1.0.zip
```

The export validates every segment index against its timestamp, maps the annotator tags to a fixed vocabulary (`TAG_MAP`), assigns dataset-wide flight IDs (`<ICAO>_<YYYY-MM-DD>_<original ID>`), sets the local meteorological season, and stores per airport-day query bounds derived from the extent of the labelled data. These bounds replace the OpenStreetMap airport outline used for the original query, which changes over time.

## Reproducing the paper results from the release

```bash
python fetch_trajectories.py --labels <release>/labels --output <release>/trajectories
python ../../../04_evaluation/paper_evaluation_zenodo.py --dataset <release> \
    --compare-with ../../../results/evaluation/paper_results/without_not_clear
```

The trajectories must be rebuilt with the locked environment of this repository (`uv.lock`), in particular `traffic` at commit `aee9ba0`.
