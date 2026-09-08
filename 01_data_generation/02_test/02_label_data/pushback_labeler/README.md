# Pushback Segment Labeler

Local web application for manually labeling pushback segments in `traffic` flight trajectories.

## Input

The labeler supports Parquet, CSV, and Pickle inputs. For the test-data workflow, use the fetched datasets from:

```text
data/test/01_fetched/<AIRPORT>/<SEASON>/<DATE>.parquet
```

At minimum, the dataset must provide flight ID, timestamp, latitude, and longitude. Groundspeed is optional and is approximated from position and time if it is not available.

## Output

Store labels outside the application directory under:

```text
data/test/02_labeled/<AIRPORT>/<SEASON>/<DATE>/
├── annotations.json
├── annotations.csv
└── settings.json
```

`annotations.json` contains the complete labels, segment boundaries, tags, and notes. `annotations.csv` provides a flattened overview, while `settings.json` stores the dataset metadata and labeler settings.

## Labeling

For each flight, the labeler supports:

- pushback segments
- push segments
- pull segments
- `no_pushback`
- `not_clear`
- optional tags and notes

The complete trajectory remains visible while segment boundaries are selected manually from the map or groundspeed plot.

## Application

The backend is implemented in `server.py`. The bundled frontend files are stored in `static/`; `frontend/` contains the editable frontend source.
