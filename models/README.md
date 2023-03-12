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
    - `alpha`: (float) Scalar for the descent derivatives
    - `scaling_method`: (string) Method used for scaling data

## Example result

```json
{
    "accuracy": 98.03210463159748,
    "model": {
        "weights": [
            0.13759739501671586,
            -0.15532505335973124,
            0.01952782972890242,
            0.2076553832868454,
            -0.17841367052533305,
            0.3561275886150834,
            -0.32791069372273274,
            0.5367065781192737,
            -0.45356964931831617,
            0.2667325934717929,
            -0.28772863585057834,
            -0.32354194738639197,
            -0.2690985386152133,
            0.4297280659706073,
            -0.05550820154182195,
            0.13502863241861068,
            0.32479349716907596,
            0.5020791207112517,
            0.21245205825077737,
            0.05687962693401111
        ],
        "bias": -0.025940128816508142
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
        "total_time": 5000,
        "period": 4,
        "iterations": 1000,
        "alpha": 0.1,
        "scaling_method": "max"
    }
}
```
