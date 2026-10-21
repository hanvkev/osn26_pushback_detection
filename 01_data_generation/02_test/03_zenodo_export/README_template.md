# Manually Labelled Aircraft Pushback Segments in ADS-B Trajectories

Kevin Hänggi (ORCID [0009-0002-4486-1796](https://orcid.org/0009-0002-4486-1796)) and
Manuel Waltert (ORCID [0000-0001-7649-6581](https://orcid.org/0000-0001-7649-6581)),
Centre for Aviation, Zurich University of Applied Sciences (ZHAW), Winterthur, Switzerland.
Contact: kevin.haenggi@zhaw.ch

Dataset DOI: [10.5281/zenodo.23548217](https://doi.org/10.5281/zenodo.23548217)

This dataset contains {n_flights} departing flights from {n_days} full days (00:00–24:00 UTC) at three
airports (Brussels Airport EBBR, Boston Logan International Airport KBOS and Aeroparque Jorge
Newbery SABE). For each flight, the
pushback manoeuvre was annotated manually. Within each pushback, the individual push and pull
phases are annotated as well ({n_segments} segments in total).

The dataset was created as the independent test set for the paper *Detection of Aircraft Pushback
Segments Using Deep Learning*, submitted to the 14th OpenSky Symposium 2026 (Madrid; proceedings in
the Journal of Open Aviation Science) and currently under review.
Code: <https://github.com/hanvkev/osn26_pushback_detection>.

The dataset contains the labels only. The trajectories are not redistributed; they are fetched
from the OpenSky historical database with the included script, which requires an
[OpenSky data access account](https://opensky-network.org/data/trino).

## Contents

```
fetch_trajectories.py        fetches the trajectories and attaches point-wise labels
labels/flights.csv           one row per flight (fetch window, fingerprints, status, tags)
labels/segments.csv          one row per annotated segment
labels/query_bounds.csv      bounding box per airport-day used for the OpenSky query
checksums.sha256             SHA-256 checksums of all files
```

## Getting the trajectories

The trajectories must be rebuilt with exactly the library versions used for labelling:
`traffic` from its GitHub repository at commit `aee9ba0` (version `2.13.post75.dev0`) and
`pyopensky 2.16`. Either use the locked environment of the code repository
(`uv sync` with its `pyproject.toml` and `uv.lock`), or install the two packages directly:

```
pip install "traffic @ git+https://github.com/xoolive/traffic.git@aee9ba062051b6b5619b5c486be2b6e24f66a1fb" "pyopensky==2.16"
```

Then configure your OpenSky credentials for pyopensky and run:

```
python fetch_trajectories.py            # all days, or e.g. --airport KBOS --date 2025-10-18
```

For each airport-day, the script queries the state vectors of all listed `icao24` addresses within
the bounding box from `query_bounds.csv` and rebuilds every flight exactly as for labelling:

1. state vectors of the flight's `icao24` and `callsign` between `first_ts` and `last_ts`
2. `traffic` `resample("1s")` (linear interpolation onto a 1 s grid)
3. `traffic` `cumulative_distance()` (adds `compute_gs` and `compute_track`)

Each rebuilt flight is compared with `n_points`, `pos_sha256` and `pb_pos_sha256` from
`flights.csv`. The result is stored in the columns `match_flight` and `match_pushback` and in
`trajectories/verification.csv`. Occasionally OpenSky returns a single raw point at the edge of a
data gap differently than at labelling time, which changes the interpolated positions within that
gap. Such flights are kept and flagged; `match_pushback` tells whether the labelled pushback part
is identical. The output is one parquet file per airport-day in `trajectories/`, with the
point-wise columns `status`, `is_pushback`, `is_push` and `is_pull`.

## Data collection

For each day, the transponder addresses (`icao24`) of all aircraft departing from or arriving at the
airport were taken from the OpenSky flight list. Their state vectors were retrieved from the OpenSky
historical database within the bounding box of the airport outline (OpenStreetMap, buffered by 0.1°).
Flights were split and identified with the [traffic](https://traffic-viz.github.io/) library
(`assign_id`) and resampled to 1 s. Departures were then identified with traffic's takeoff detection
(`takeoff(method="track_based")`); only these flights are included.
The airport outline from OpenStreetMap changes over time. Since the
positions are interpolated linearly, the labelled points span exactly the raw positions that were
inside the original query area; `query_bounds.csv` therefore gives the extent of the labelled data
per airport-day, which selects the same raw state vectors independently of OpenStreetMap.

The days were chosen to cover different seasons. `season` is the meteorological season at the
airport (December–February winter on the northern hemisphere), i.e. reversed for SABE on the
southern hemisphere (e.g. 2025-08-08 is winter at SABE).

## Statistics

{statistics_table}

## File formats

All timestamps are in UTC.

### trajectories/*.parquet (created by fetch_trajectories.py)

| Column | Description |
|---|---|
| `flight_id` | Flight identifier `<ICAO>_<YYYY-MM-DD>_<callsign>_<n>`, unique across the dataset |
| `point_idx` | Position of the point within the flight (0-based, ordered by `timestamp`) |
| `timestamp` | Time on the 1 s grid |
| `icao24`, `callsign`, `squawk` | Transponder address, callsign and squawk |
| `latitude`, `longitude` | WGS84 position in degrees |
| `altitude`, `geoaltitude` | Barometric and geometric altitude in ft (mostly empty on the ground) |
| `vertical_rate` | ft/min |
| `groundspeed`, `track` | Groundspeed (kt) and track (deg) as reported by OpenSky |
| `compute_gs`, `compute_track` | Groundspeed (kt) and track (deg) computed from consecutive positions (traffic `cumulative_distance`). These were used for labelling. |
| `onground` | Surface flag reported by the transponder (1 = on ground, 0 = airborne; a few fractional values stem from interpolation) |
| `status`, `is_pushback`, `is_push`, `is_pull` | Flight status and point-wise labels from `labels/` |

### labels/flights.csv

| Column | Description |
|---|---|
| `flight_id` | Key to the trajectories |
| `source_flight_id` | Original ID assigned by traffic (`<callsign>_<n>`, only unique within a day) |
| `airport`, `date`, `season` | Airport ICAO code, day (YYYY-MM-DD) and meteorological season at the airport |
| `icao24`, `callsign` | Transponder address and callsign |
| `first_ts`, `last_ts` | First and last point of the flight (fetch window) |
| `n_points` | Number of points after resampling |
| `pos_sha256` | SHA-256 of all positions of the flight (`lat,lon` with 6 decimals, one line per point) |
| `pb_pos_sha256` | Same for the points from the first pushback start to the last pushback end (empty without pushback) |
| `status` | `labelled` (pushback annotated), `no_pushback` (no pushback observed), `not_clear` (presence or extent of a pushback could not be determined with sufficient confidence) |
| `n_pushback`, `n_push`, `n_pull` | Number of annotated segments per type |
| `tags` | Semicolon-separated tags (see below) |

### labels/segments.csv

| Column | Description |
|---|---|
| `flight_id` | Key to the trajectories |
| `segment_type` | `pushback` (complete manoeuvre), `push` (aircraft pushed backwards), `pull` (aircraft pulled forward by the tug) |
| `segment_no` | Running number per flight and segment type |
| `start_ts`, `end_ts` | First and last point of the segment (inclusive) |
| `start_idx`, `end_idx` | The same points as `point_idx` in the fetched trajectories |
| `duration_s` | `end_ts - start_ts` in seconds |

`push` and `pull` segments lie within the `pushback` segment. They do not have to cover it
completely, e.g. a stop between pushing and pulling belongs to neither.

## Labelling

All flights were labelled by a single annotator with a custom labelling tool, which is part of the
code repository (`01_data_generation/02_test/02_label_data/pushback_labeler`). The tool shows the
complete trajectory on a map together with the groundspeed (`compute_gs`) over time; segment
boundaries were selected manually on individual points, from the map or the groundspeed plot.
For each flight, the annotator assigned the status (`labelled`, `no_pushback`, `not_clear`), the
pushback segment, the push and pull phases within it, and optional tags.

- **Pushback.** A pushback segment is the sequence of trajectory points belonging to the pushback
  manoeuvre. It extends from the start to the end of the associated aircraft movement and includes
  intermediate stationary points within multi-stage or push-pull manoeuvres. Stationary points before
  or after the manoeuvre, including those associated with tug coupling or decoupling, are excluded.
  At most one complete pushback is assigned to each departure. If the trajectory begins after the
  pushback had already started (e.g. late transponder activation), the segment starts at the first
  available point.
- **Push and pull.** Within a pushback, the phases in which the aircraft was pushed backwards (push)
  or pulled forwards by the tug (pull) were annotated where identifiable from the trajectory and its
  surroundings.
- **Not clear.** Flights for which the presence or the temporal extent of a pushback could not be
  determined with sufficient confidence are labelled `not_clear`. Pushback segments may still be
  present for such flights where the annotator marked a possible manoeuvre.
- **Interpolated positions.** Stretches of the trajectory filled by the 1 s resampling (straight
  lines at constant speed across data gaps) were judged case by case, i.e. whether they belong to the
  pushback was decided per flight; such cases are partly marked with tags.

All annotations were made by one person, so no inter-annotator agreement is available, and the
interpretation of individual cases and exact boundaries may have evolved with annotation experience.

### Tags

Tags are informal notes of the annotator to mark interesting or difficult cases during labelling. They were not assigned systematically and are neither complete nor validated.

{tag_table}

## License and attribution

The labels and documentation of this dataset are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The trajectories are not part of this dataset. They are retrieved by the user from the historical
database of the [OpenSky Network](https://opensky-network.org/) and are subject to its terms of use.
When using this dataset, please cite this dataset, the paper (once
published), and

- M. Schäfer, M. Strohmeier, V. Lenders, I. Martinovic, M. Wilhelm. *Bringing Up OpenSky: A
  Large-scale ADS-B Sensor Network for Research.* IPSN 2014.
- X. Olive. *traffic, a toolbox for processing and analysing air traffic data.* Journal of Open
  Source Software 4(39), 2019.
