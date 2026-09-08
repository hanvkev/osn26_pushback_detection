# Pushback Detection from ADS-B Data

This repository contains the data preparation, model optimisation, final model, evaluation, and plotting code for a paper prepared for the [14th OpenSky Symposium 2026](https://symposium.opensky-network.org/) in Madrid.

The project focuses on detecting aircraft pushback segments from ADS-B trajectories using a Temporal Convolutional Network (TCN).

## Repository structure

- `01_data_generation/` – generation and labelling of training, validation, and test data
- `02_feature_hyperparameter_optimization/` – feature and hyperparameter optimisation
- `03_final_model/` – final model training and inference pipeline
- `04_evaluation/` – evaluation on manually labelled test trajectories
- `05_plots/` – figures used in the paper

Generated datasets and outputs are excluded from version control where appropriate.