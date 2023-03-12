# Saving trained models

In order to avoid restarting the training every time from scratch, the results and configuration for a training and test session will be stored in this folder with `json` format.

Those models can be loaded in main script by appending their relative path as a command-line argument, for example :

`python3 main.py model_2023_03_12_16:39:25.json`

## Naming

The pretrained model will be named according to the timestamp of their testing, formatted `model_%Y_%m_%d_%:%M:%S.json`

## JSON structure

- `accuracy`: (number) Accuracy of the model computed from testing step
- `model`: (dictionnary) Model's weights and bias
    - `weights`: (list[float]) Model's wheights
    - `bias`: (float) Model's bias
- `config`: (dictionnary) Configuration used for training and testing the model

    - `csv_path`: (string) Path to the csv dataset
    - `target_label`: (string) Label of the target column
    - `exclude_labels`: ([string]) Labels of the excluded columns
    - `total_time`: (int) Total studied period
    - `period`: (int) Number of rows to form one input
    - `iterations`: (int) Max number of iterations for gradient descent
    - `convergence_threshold`: (float) threshold to determine if convergence is met
    - `alpha`: (float) Scalar for the descent derivatives
    - `scaling_method`: (string) Method used for scaling data

## Example result

```json
{
    "accuracy": 98.97887758135148,
    "model": {
        "weights": [
            -0.1988854751413387,
            -0.10154046495976074,
            -0.010690176192727413,
            -0.2997025653204071,
            -0.006933326794411775,
            0.25350926913677313,
            0.018607274303323,
            -0.0061027909325161635,
            0.25277532264423513,
            -0.04202455717108864,
            0.5315169980281808,
            -0.2838240530005194,
            -0.18657472643470888,
            0.25997144147022994,
            0.0007701811243179069,
            0.01726992882632942,
            0.532002498216597,
            -0.11976741485266931,
            0.3396799808732589,
            0.047598237743463785
        ],
        "bias": 0.0016376338018652583
    },
    "config": {
        "csv_path": "spy.csv",
        "target_label": "Close",
        "exclude_labels": [
            "Date",
            "day",
            "week",
            "weekday",
            "year"
        ],
        "total_time": 2500,
        "period": 4,
        "iterations": 10000,
        "convergence_threshold": 1e-8,
        "alpha": 0.1,
        "scaling_method": "max"
    }
}
```
