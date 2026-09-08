# Data Generation

This directory contains the workflows used to generate the training, validation, and test datasets.

## Training and validation data

`01_train_validation/` contains the steps for:

1. fetching raw trajectory and flight-history data,
2. generating rule-based pushback labels,
3. matching trajectory and flight-history data, and
4. preparing the final airport datasets.

The final datasets are stored under:

```text
data/train_validation/05_final/<AIRPORT>/
```

## Test data

`02_test/` contains the data-fetching workflow and the manual pushback labelling application.

Fetched and labelled test data are stored under:

```text
data/test/01_fetched/
data/test/02_labeled/
```
