# Pushback Detection from ADS-B Data

This repository contains the data preparation, model optimisation, final model, evaluation, and plotting code for a paper prepared for the [14th OpenSky Symposium 2026](https://symposium.opensky-network.org/) in Madrid.

The project focuses on detecting aircraft pushback segments from ADS-B trajectories using a Temporal Convolutional Network (TCN).

## Repository structure

- `01_data_generation/` – generation and labelling of training, validation, and test data
- `02_feature_hyperparameter_optimization/` – feature and hyperparameter optimisation
- `03_final_model/` – final model training and inference pipeline
- `04_evaluation/` – evaluation on manually labelled test trajectories
- `05_plots/` – figures used in the paper

## Test dataset

The manually labelled test dataset is available on Zenodo: https://doi.org/10.5281/zenodo.23548217. It contains the labels and a script to rebuild the trajectories from the OpenSky historical database; see `01_data_generation/02_test/03_zenodo_export/`.

Generated datasets and outputs are excluded from version control where appropriate.