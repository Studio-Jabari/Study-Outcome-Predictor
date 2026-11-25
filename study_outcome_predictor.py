#!/usr/bin/env python3
"""Educational logistic-regression demonstration for student study outcomes.

The model estimates the probability of a binary outcome (pass/fail) from a
single feature: study hours. It is intended for learning, not real-world
student assessment.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterable, Sequence

TrainingRecord = tuple[float, int]
TrainingData = Sequence[TrainingRecord]

# Original 630 records
TRAINING_DATA: tuple[TrainingRecord, ...] = ((0.0, 0), (0.1, 0), (0.2, 0), (0.2, 0), (0.3, 0), (0.4, 0), (0.5, 0), (0.5, 0), (0.6, 0), (0.6, 0), (0.7, 0), (0.7, 0), (0.8, 0), (0.8, 0), (0.9, 0), (0.9, 0), (1.0, 0), (1.0, 0), (1.0, 0), (1.1, 0), (1.1, 0), (1.2, 0), (1.2, 0), (1.2, 0), (1.3, 0), (1.3, 0), (1.4, 0), (1.4, 0), (1.5, 0), (1.5, 0), (1.5, 0), (1.5, 0), (1.6, 0), (1.6, 0), (1.7, 0), (1.7, 0), (1.7, 0), (1.8, 0), (1.8, 0), (1.8, 0), (1.9, 0), (1.9, 0), (1.9, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 1), (2.0, 0), (2.1, 0), (2.1, 0), (2.2, 0), (2.2, 0), (2.2, 0), (2.3, 0), (2.3, 0), (2.3, 0), (2.4, 0), (2.4, 0), (2.4, 0), (2.4, 0), (2.5, 0), (2.5, 0), (2.5, 0), (2.5, 0), (2.5, 1), (2.5, 0), (2.6, 0), (2.6, 0), (2.6, 0), (2.7, 0), (2.7, 0), (2.7, 0), (2.7, 0), (2.8, 0), (2.8, 0), (2.8, 0), (2.8, 0), (2.8, 1), (2.9, 0), (2.9, 0), (2.9, 0), (2.9, 0), (3.0, 0), (3.0, 0), (3.0, 0), (3.0, 1), (3.0, 0), (3.0, 0), (3.1, 0), (3.1, 0), (3.1, 0), (3.2, 0), (3.2, 0), (3.2, 0), (3.2, 0), (3.3, 0), (3.3, 0), (3.3, 1), (3.4, 0), (3.4, 0), (3.4, 0), (3.4, 0), (3.5, 0), (3.5, 0), (3.5, 1), (3.5, 0), (3.5, 1), (3.5, 0), (3.6, 0), (3.6, 0), (3.6, 0), (3.7, 0), (3.7, 0), (3.7, 1), (3.7, 0), (3.8, 0), (3.8, 0), (3.8, 0), (3.8, 1), (3.8, 0), (3.9, 0), (3.9, 0), (3.9, 0), (3.9, 1), (3.9, 0), (4.0, 0), (4.0, 0), (4.0, 1), (4.0, 0), (4.0, 1), (4.0, 0), (4.1, 0), (4.1, 0), (4.1, 0), (4.1, 1), (4.1, 0), (4.2, 0), (4.2, 0), (4.2, 1), (4.2, 0), (4.2, 1), (4.3, 0), (4.3, 0), (4.3, 0), (4.3, 1), (4.3, 1), (4.4, 0), (4.4, 0), (4.4, 1), (4.4, 0), (4.4, 1), (4.4, 0), (4.5, 0), (4.5, 0), (4.5, 1), (4.5, 1), (4.5, 0), (4.5, 1), (4.6, 0), (4.6, 1), (4.6, 0), (4.6, 1), (4.6, 0), (4.7, 0), (4.7, 1), (4.7, 0), (4.7, 1), (4.7, 1), (4.8, 0), (4.8, 1), (4.8, 0), (4.8, 1), (4.8, 1), (4.9, 0), (4.9, 1), (4.9, 1), (4.9, 0), (4.9, 1), (5.0, 0), (5.0, 1), (5.0, 0), (5.0, 1), (5.0, 1), (5.0, 0), (5.0, 1), (5.0, 1), (5.1, 0), (5.1, 1), (5.1, 1), (5.1, 0), (5.1, 1), (5.2, 0), (5.2, 1), (5.2, 1), (5.2, 1), (5.2, 0), (5.3, 1), (5.3, 0), (5.3, 1), (5.3, 1), (5.3, 1), (5.4, 0), (5.4, 1), (5.4, 1), (5.4, 1), (5.4, 0), (5.5, 0), (5.5, 1), (5.5, 1), (5.5, 0), (5.5, 1), (5.5, 1), (5.5, 1), (5.5, 0), (5.5, 1), (5.5, 1), (5.5, 1), (5.5, 0), (5.6, 1), (5.6, 0), (5.6, 1), (5.6, 1), (5.6, 1), (5.7, 0), (5.7, 1), (5.7, 1), (5.7, 1), (5.7, 0), (5.8, 1), (5.8, 1), (5.8, 0), (5.8, 1), (5.8, 1), (5.9, 1), (5.9, 1), (5.9, 1), (5.9, 0), (5.9, 1), (6.0, 1), (6.0, 0), (6.0, 1), (6.0, 1), (6.0, 1), (6.0, 1), (6.0, 0), (6.0, 1), (6.1, 1), (6.1, 1), (6.1, 1), (6.1, 1), (6.2, 1), (6.2, 1), (6.2, 0), (6.2, 1), (6.3, 1), (6.3, 1), (6.3, 1), (6.3, 1), (6.4, 1), (6.4, 1), (6.4, 1), (6.4, 1), (6.4, 0), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 0), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 1), (6.5, 0), (6.5, 1), (6.6, 1), (6.6, 1), (6.6, 1), (6.7, 1), (6.7, 1), (6.7, 1), (6.8, 1), (6.8, 1), (6.8, 0), (6.9, 1), (6.9, 1), (6.9, 1), (7.0, 1), (7.0, 1), (7.0, 1), (7.0, 1), (7.0, 1), (7.0, 0), (7.0, 1), (7.1, 1), (7.1, 1), (7.1, 1), (7.2, 1), (7.2, 1), (7.2, 1), (7.3, 1), (7.3, 1), (7.3, 1), (7.4, 1), (7.4, 1), (7.4, 1), (7.5, 1), (7.5, 1), (7.5, 1), (7.5, 1), (7.5, 1), (7.6, 1), (7.6, 1), (7.6, 1), (7.7, 1), (7.7, 1), (7.7, 1), (7.8, 1), (7.8, 1), (7.8, 1), (7.9, 1), (7.9, 1), (7.9, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.0, 1), (8.1, 1), (8.1, 1), (8.2, 1), (8.2, 1), (8.3, 1), (8.3, 1), (8.4, 1), (8.4, 1), (8.5, 1), (8.5, 1), (8.5, 1), (8.6, 1), (8.6, 1), (8.7, 1), (8.7, 1), (8.8, 1), (8.8, 1), (8.9, 1), (8.9, 1), (9.0, 1), (9.0, 1), (9.0, 1), (9.1, 1), (9.1, 1), (9.2, 1), (9.2, 1), (9.3, 1), (9.3, 1), (9.4, 1), (9.4, 1), (9.5, 1), (9.5, 1), (9.5, 1), (9.6, 1), (9.6, 1), (9.7, 1), (9.7, 1), (9.8, 1), (9.8, 1), (9.9, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.0, 1), (10.2, 1), (10.4, 1), (10.5, 1), (10.6, 1), (10.8, 1), (11.0, 1), (11.0, 1), (11.2, 1), (11.4, 1), (11.5, 1), (11.6, 1), (11.8, 1), (12.0, 1), (12.0, 1), (12.2, 1), (12.4, 1), (12.5, 1), (12.6, 1), (12.8, 1), (13.0, 1), (13.0, 1), (13.2, 1), (13.4, 1), (13.5, 1), (13.6, 1), (13.8, 1), (14.0, 1), (14.0, 1), (14.2, 1), (14.4, 1), (14.5, 1), (14.6, 1), (14.8, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (0.5, 0), (0.8, 0), (1.2, 0), (1.6, 0), (2.2, 0), (2.4, 0), (2.8, 0), (3.1, 0), (3.3, 0), (3.6, 0), (3.9, 0), (4.1, 0), (4.3, 0), (4.4, 1), (4.6, 1), (4.7, 0), (4.8, 1), (4.9, 1), (5.1, 0), (5.2, 1), (5.3, 1), (5.4, 1), (5.6, 1), (5.8, 1), (6.1, 1), (6.3, 1), (6.6, 1), (6.9, 1), (7.2, 1), (7.6, 1), (8.1, 1), (8.6, 1), (9.2, 1), (9.8, 1), (10.5, 1), (11.2, 1), (12.0, 1), (13.0, 1), (14.0, 1), (15.0, 1), (0.3, 0), (0.7, 0), (1.1, 0), (1.4, 0), (1.9, 0), (2.3, 0), (2.7, 0), (3.0, 0), (3.2, 0), (3.5, 1), (3.8, 0), (4.0, 1), (4.2, 0), (4.5, 1), (4.6, 0), (4.7, 1), (4.9, 0), (5.0, 1), (5.2, 0), (5.3, 1), (5.5, 1), (5.7, 1), (5.9, 1), (6.2, 1), (6.4, 1), (6.8, 1), (7.1, 1), (7.5, 1), (8.0, 1), (8.5, 1), (9.0, 1), (9.6, 1), (10.2, 1), (11.0, 1), (11.8, 1), (12.5, 1), (13.5, 1), (14.5, 1), (15.0, 1), (15.0, 1), (0.4, 0), (0.9, 0), (1.3, 0), (1.8, 0), (2.1, 0), (2.6, 0), (2.9, 0), (3.4, 0), (3.7, 0), (4.0, 0), (4.3, 1), (4.5, 0), (4.7, 0), (4.8, 1), (5.0, 0), (5.1, 1), (5.3, 0), (5.4, 1), (5.6, 0), (5.8, 1), (6.0, 1), (6.3, 1), (6.5, 1), (6.7, 1), (7.0, 1), (7.4, 1), (7.8, 1), (8.3, 1), (8.8, 1), (9.4, 1), (10.0, 1), (10.6, 1), (11.4, 1), (12.2, 1), (13.0, 1), (14.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (0.6, 0), (1.0, 0), (1.5, 0), (2.0, 0), (2.5, 0), (3.0, 0), (3.5, 0), (3.8, 1), (4.1, 1), (4.4, 0), (4.6, 1), (4.9, 1), (5.1, 1), (5.2, 0), (5.4, 1), (5.5, 1), (5.7, 0), (5.9, 1), (6.1, 1), (6.4, 1), (6.6, 1), (7.0, 1), (7.3, 1), (7.7, 1), (8.2, 1), (8.7, 1), (9.3, 1), (10.0, 1), (10.8, 1), (11.6, 1), (12.4, 1), (13.2, 1), (14.2, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (0.2, 0), (0.7, 0), (1.2, 0), (1.7, 0), (2.3, 0), (2.8, 0), (3.2, 0), (3.6, 0), (4.0, 0), (4.2, 1), (4.5, 1), (4.7, 1), (4.9, 1), (5.0, 0), (5.2, 1), (5.3, 1), (5.5, 0), (5.6, 1), (5.8, 1), (6.0, 1), (6.3, 1), (6.5, 1), (6.8, 1), (7.2, 1), (7.6, 1), (8.0, 1), (8.5, 1), (9.1, 1), (9.7, 1), (10.4, 1), (11.1, 1), (11.9, 1), (12.8, 1), (13.8, 1), (14.8, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1), (15.0, 1))

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS = 50_000
CLASSIFICATION_THRESHOLD = 0.5


def sigmoid(value: float) -> float:
    """Return the logistic sigmoid using numerically stable branches."""
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def binary_cross_entropy(probability: float, actual: int) -> float:
    """Compute binary cross-entropy for one record, clipping at machine-safe bounds."""
    epsilon = 1e-15
    probability = min(max(probability, epsilon), 1.0 - epsilon)
    return -(actual * math.log(probability) + (1 - actual) * math.log1p(-probability))


def validate_training_data(records: Iterable[TrainingRecord]) -> tuple[TrainingRecord, ...]:
    """Validate and normalize records; labels must be binary and hours nonnegative."""
    validated: list[TrainingRecord] = []
    for index, record in enumerate(records):
        if len(record) != 2:
            raise ValueError(f"Record {index} must contain (hours, label).")
        hours, label = record
        if not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0:
            raise ValueError(f"Record {index} has invalid study hours.")
        if label not in (0, 1):
            raise ValueError(f"Record {index} label must be 0 or 1.")
        validated.append((float(hours), int(label)))
    if not validated:
        raise ValueError("Training data must not be empty.")
    return tuple(validated)


def clear_screen() -> None:
    """Clear the terminal or, when available, the IPython notebook output."""
    try:
        from IPython.display import clear_output  # Optional dependency
    except ImportError:
        print("\\033[2J\\033[H", end="")
    else:
        clear_output(wait=True)


def draw_live_plot(epoch: int, max_epochs: int, loss: float, slope: float, intercept: float) -> None:
    """Render the model's probability curve as an ASCII chart."""
    clear_screen()
    plot_width = 45
    print("=" * 55)
    print(f" TRAINING MODEL: Epoch {epoch}/{max_epochs} | BCE Loss: {loss:.4f}")
    print("=" * 55)
    print(" Pass (1.0) ┼" + " " * plot_width)

    for row in range(10, -1, -1):
        target_probability = row / 10.0
        line = [" "] * plot_width
        for column in range(plot_width):
            hours = (column / plot_width) * 12.0
            probability = sigmoid(slope * hours + intercept)
            if abs(probability - target_probability) < 0.05:
                line[column] = "█"
        if row == 5:
            prefix = "Prob (0.5) ┼"
        elif row == 0:
            prefix = " Fail (0.0) ┼"
        else:
            prefix = "           │"
        print(prefix + "".join(line))

    print("           └" + "─" * plot_width)
    print(" Hours:     0.0        3.0        6.0        9.0       12.0")
    print(f"\\n Model: P(pass) = sigmoid({slope:.3f} × hours + {intercept:.3f})")


def train_model_with_visualization(
    training_data: TrainingData = TRAINING_DATA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = DEFAULT_EPOCHS,
    visualize: bool = True,
    update_interval: int = 100,
    animation_delay: float = 0.04,
    plot_callback: Callable[[int, int, float, float, float], None] = draw_live_plot,
) -> tuple[float, float]:
    """Train one-feature logistic regression using batch gradient descent.

    The gradient is the standard BCE/logistic-regression gradient. Visualization
    is optional so the training routine can also be used in scripts and tests.
    Returns the learned (slope, intercept).
    """
    records = validate_training_data(training_data)
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be a finite positive number.")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if update_interval < 1:
        raise ValueError("update_interval must be positive.")
    slope = intercept = 0.0
    record_count = len(records)

    for epoch in range(1, epochs + 1):
        slope_gradient = intercept_gradient = total_loss = 0.0
        for hours, actual in records:
            predicted = sigmoid(slope * hours + intercept)
            error = predicted - actual
            slope_gradient += error * hours
            intercept_gradient += error
            total_loss += binary_cross_entropy(predicted, actual)

        slope -= learning_rate * slope_gradient / record_count
        intercept -= learning_rate * intercept_gradient / record_count

        if visualize and (epoch == 1 or epoch % update_interval == 0 or epoch == epochs):
            plot_callback(epoch, epochs, total_loss / record_count, slope, intercept)
            if animation_delay > 0:
                time.sleep(animation_delay)
    return slope, intercept


def get_user_input() -> float:
    """Prompt for study hours in the range 0–24, retrying invalid entries."""
    while True:
        try:
            hours = float(input("\\nEnter study hours to test the trained model: "))
            if not math.isfinite(hours) or not 0 <= hours <= 24:
                print("Please enter a number between 0 and 24.")
                continue
            return hours
        except ValueError:
            print("Invalid number. Please enter a numeric value.")


def predict_pass_fail(hours: float, slope: float, intercept: float) -> tuple[str, float]:
    """Return the classification label and estimated pass probability."""
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0:
        raise ValueError("hours must be a finite, nonnegative number.")
    probability = sigmoid(slope * hours + intercept)
    status = "Likely to pass" if probability >= CLASSIFICATION_THRESHOLD else "At risk of failing"
    return status, probability


def main() -> None:
    """Train the model, prompt for a test input, and display the prediction."""
    slope, intercept = train_model_with_visualization()
    user_hours = get_user_input()
    status, probability = predict_pass_fail(user_hours, slope, intercept)
    print("\\n" + "=" * 35)
    print(f" Hours Studied: {user_hours:g} hours")
    print(f" Result:        Student is {status}")
    print(f" Confidence:    {probability * 100:.1f}%")
    print("=" * 35)


if __name__ == "__main__":
    main()
