# Study Outcome Predictor

An educational, dependency-free Python demonstration of logistic regression.

This program estimates a student's probability of passing from one input feature: reported study hours. It trains a logistic regression model from a hand-constructed dataset, using batch gradient descent, and displays a live ASCII probability curve during training.

> **Educational disclaimer:** This dataset is synthetic/hand-constructed and the model has not been validated for real student outcomes. Its probabilities are not evidence-based assessments and should not be used for academic decisions.

## Features

- Logistic regression implemented from scratch (no ML libraries).
- 630 original training records preserved in their original order.
- Batch gradient descent (50,000 epochs; learning rate 0.1 by default).
- Stable sigmoid calculation.
- Binary cross-entropy loss tracking.
- Live ASCII visualization, rendered every 100 epochs.
- Optional IPython notebook clearing; IPython is not required.
- Input and training-data validation.

## Requirements

- Python 3.9 or newer.
- No required third-party packages. If IPython is installed, notebook output is cleared using its display helper; otherwise the program clears a terminal screen.

## Run

Clone/download the repository, then run:

```bash
python study_outcome_predictor.py
```

The program trains the model, prompts for study hours (0–24), and prints the predicted class and estimated probability.

## How it works

For study hours `x`, the model computes a linear score:

`z = slope * x + intercept`

The sigmoid converts the score to a probability:

`P(pass) = 1 / (1 + exp(-z))`

Training updates the slope and intercept using batch gradient descent and the Binary Cross-Entropy (BCE) objective. The gradient for this logistic-regression/BCE pairing is based on `prediction - actual`; the original implementation's parameter-update equations are retained. The visualization reports average BCE loss (rather than Mean Squared Error).

Predictions use a 0.5 threshold: probabilities at least 0.5 are labeled **Likely to pass**, otherwise **At risk of failing**.

## Dataset

The training data consists of 630 `(study_hours, outcome)` records, where `outcome` is `0` (failed) or `1` (passed). Study hours in the training records range from 0 to 15. The records are preserved as originally supplied; this is not a curated or validated research dataset.

## Code structure

- `TRAINING_DATA`: the original training records.
- `sigmoid()`: numerically stable logistic function.
- `binary_cross_entropy()`: BCE loss for one example.
- `validate_training_data()`: validates labels and values.
- `train_model_with_visualization()`: trains the model and optionally visualizes.
- `draw_live_plot()`: ASCII visualization.
- `predict_pass_fail()`: inference.
- `main()`: command-line flow.

Training and plotting can be imported and called separately. For example:

```python
from study_outcome_predictor import train_model_with_visualization, predict_pass_fail

slope, intercept = train_model_with_visualization(epochs=1000, visualize=False)
label, probability = predict_pass_fail(6.0, slope, intercept)
print(label, probability)
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
