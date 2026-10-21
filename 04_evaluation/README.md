# Evaluation

This directory contains the evaluation of the final pushback-detection pipeline on manually labelled test trajectories.

The PRD-TCN pipeline is compared with the rule-based `parking_area` and `parking_position` methods.

The evaluation includes flight-level detection metrics, segment-level metrics at different IoU thresholds, and temporal segment-quality measures.

Generated evaluation results are stored under:

```text
results/evaluation/
```

`paper_evaluation_zenodo.py` runs the same evaluation on the public Zenodo release of the test dataset (see `01_data_generation/02_test/03_zenodo_export/`). It executes the code cells of `paper_evaluation.ipynb` unchanged and only replaces the data source, so both produce identical paper tables:

```bash
python paper_evaluation_zenodo.py --dataset <release folder with trajectories/> \
    --compare-with ../results/evaluation/paper_results/without_not_clear
```

Its outputs are written to `results/evaluation_zenodo/`.
