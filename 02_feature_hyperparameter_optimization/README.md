# PRD-TCN Feature Configuration and Hyperparameter Optimization

Bayesian optimization pipeline for jointly selecting the feature configuration and model hyperparameters of the pointwise TCN pushback detector.

Each configuration is evaluated using leave-one-airport-out (LOAO) cross-validation across LSZH, ESSA, RJTT, and YPPH.

## Structure

```text id="q1u2l4"
02_feature_hyperparameter_optimization/
├── README.md
├── create_sweep.py
├── sweep.py
├── sweep_30runs.yaml
├── run_agent.sh
├── run_wandb_sweep.slurm
│
└── hp_optimization/
    ├── __init__.py
    ├── experiment.py
    ├── config.py
    ├── runtime.py
    ├── features.py
    ├── data.py
    ├── model.py
    ├── training.py
    ├── metrics.py
    ├── postprocessing.py
    ├── evaluation.py
    ├── artifacts.py
    └── utils.py
```

`hp_optimization/` contains the complete optimization pipeline, including feature generation, LOAO training, scaling, prediction, postprocessing optimization, evaluation, and artifact generation.

## Optimization

The optimization jointly considers:

* feature variant
* track-window configuration
* learning rate
* dropout
* positive-class weight scaling

The available feature variants combine:

* computed ground speed
* ground-speed difference
* track difference
* circular track-change features

The circular track-change variants additionally optimize the track-window configuration.

The TCN architecture itself remains fixed during this stage.

## LOAO Evaluation

Each sampled configuration is evaluated using four leave-one-airport-out folds.

For each fold, one airport is held out while the remaining airports are used for training and validation.

The scaler and class weights are fitted only on the corresponding training data.

After all folds are complete, the held-out predictions are combined into one out-of-fold prediction set covering all four airports.

This ensures that each airport is evaluated using predictions from a model that was not trained on that airport.

## Optimization Objective

The Bayesian optimizer maximizes:

```text id="a9qp0a"
0.50 * macro flight balanced accuracy
+ 0.20 * macro segment F1 at IoU >= 0.50
+ 0.30 * macro segment F1 at IoU >= 0.75
```

The objective therefore combines flight-level detection performance with segment-level localization quality.

## Core Postprocessing

After the four LOAO folds have produced out-of-fold probabilities, a global core-postprocessing search is performed over:

* core probability threshold
* maximum interruption duration

Smoothing and boundary expansion are not included in this optimization stage.

The resulting feature and hyperparameter configuration is subsequently used for the final all-airports model in:

```text id="k98fqv"
03_final_model/
```
