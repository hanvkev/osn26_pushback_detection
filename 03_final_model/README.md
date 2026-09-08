# Final Model

Training and inference pipeline for the final pointwise TCN pushback detector.

## Structure

```text
03_final_model/
├── README.md
├── train_final_model.py
├── run_final_model.slurm
├── pushback_prediction.py
├── example.ipynb
│
├── final_model_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── runtime.py
│   ├── features.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── metrics.py
│   ├── postprocessing.py
│   ├── evaluation.py
│   └── utils.py
│
└── model/
    ├── final_model.pt
    └── final_scaler.pkl
```

`final_model_pipeline/` contains the code required to train and validate the final model.

`pushback_prediction.py` contains the standalone inference pipeline.

`example.ipynb` demonstrates how to apply the trained model to OpenSky trajectories.

## Final Training

The final model is trained on the prepared datasets from LSZH, ESSA, RJTT, and YPPH.

Each airport is split independently into training and validation flights. The training subsets are pooled to train one all-airports model, while the validation subsets are used for early stopping and final core-postprocessing optimization.

The selected feature configuration and model hyperparameters originate from the preceding feature and hyperparameter optimization stage.

After training, the core-postprocessing parameters are optimized globally on the validation data. The selected parameters are stored together with the trained model in `final_model.pt`.

## Inference

The selected model artifacts are stored in:

```text
model/
├── final_model.pt
└── final_scaler.pkl
```

The final inference pipeline can be imported with:

```python
from pushback_prediction import predict_pushback
```

and applied to a `traffic` `Flight` object:

```python
pushback = predict_pushback(
    flight,
    airport="LSZH",
)
```

The function returns the detected pushback segment as a `Flight`, or `None` if no pushback is detected.

The inference pipeline includes:

* trajectory preprocessing
* feature generation
* feature scaling
* TCN inference
* optimized core detection
* fixed boundary expansion

## Example

`example.ipynb` demonstrates the complete workflow using OpenSky data from LSZH, including preprocessing, departure filtering, pushback prediction, and visualization.
