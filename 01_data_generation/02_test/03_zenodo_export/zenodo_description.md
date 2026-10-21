<!-- Text for the "Description" field of the Zenodo record (the detailed documentation is README.md in the archive). -->

<!--
Zenodo metadata:
  Resource type: Dataset
  Title:         Manually Labelled Aircraft Pushback Segments in ADS-B Trajectories
  Creators:      Hänggi, Kevin  (ORCID 0009-0002-4486-1796)  Centre for Aviation, Zurich University of Applied Sciences
                 Waltert, Manuel (ORCID 0000-0001-7649-6581)  Centre for Aviation, Zurich University of Applied Sciences
  Version:       1.0
  License:       Creative Commons Attribution 4.0 International (CC BY 4.0)
  Keywords:      pushback, ADS-B, OpenSky Network, airport surface movement, ground operations, trajectory labelling
  Related:       https://github.com/hanvkev/osn26_pushback_detection  (is supplemented by, Software)
  DOI:           10.5281/zenodo.23548217 (reserved)
-->

This dataset contains manually labelled aircraft pushback manoeuvres for 3,167 departures at three airports (Brussels Airport EBBR, Boston Logan International Airport KBOS and Aeroparque Jorge Newbery SABE), observed on 16 full days between January 2025 and June 2026 in different seasons (all four seasons at EBBR and SABE, three at KBOS). For every flight, a single annotator marked whether a pushback occurred and, if so, its start and end; within each pushback, the individual push and pull phases are annotated where identifiable. In total, 2,227 flights contain a labelled pushback (4,933 annotated segments), 917 flights have no pushback and 23 flights are marked as not clear.

The dataset was created as the independent test set for the paper *Detection of Aircraft Pushback Segments Using Deep Learning*, submitted to the 14th OpenSky Symposium 2026 in Madrid (proceedings in the Journal of Open Aviation Science) and currently under review. When using this data in a scientific publication, please cite this dataset and, once published, the paper.

**Trajectories.** The ADS-B trajectories are not redistributed. They are fetched from the historical database of the OpenSky Network with the included script, which requires an OpenSky data access account. The script rebuilds every flight exactly as it was labelled and verifies it against fingerprints of the labelled positions, so the point indices of the labels refer to the same points.

The archive contains:

- **labels/flights.csv**: one row per flight with airport, date, season, transponder address, callsign, time window, number of points, fingerprints, label status and annotator tags.
- **labels/segments.csv**: one row per annotated segment (pushback, push or pull) with start and end timestamps and point indices.
- **labels/query_bounds.csv**: bounding box per airport-day for the OpenSky query.
- **fetch_trajectories.py**: fetches the trajectories and writes one file per airport-day with point-wise labels.
- **README.md**: data collection, labelling, file formats and column definitions.

To obtain the trajectories, run **fetch_trajectories.py** after configuring your OpenSky credentials; the required library versions are listed in **README.md**.

The code for labelling, model training and evaluation is available at https://github.com/hanvkev/osn26_pushback_detection.

The trajectories are derived from data of the OpenSky Network (Schäfer et al., IPSN 2014) and processed with the traffic library (Olive, JOSS 2019).
