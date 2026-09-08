# Evaluation

This directory contains the evaluation of the final pushback-detection pipeline on manually labelled test trajectories.

The PRD-TCN pipeline is compared with the rule-based `parking_area` and `parking_position` methods.

The evaluation includes flight-level detection metrics, segment-level metrics at different IoU thresholds, and temporal segment-quality measures.

Generated evaluation results are stored under:

```text
results/evaluation/
```
