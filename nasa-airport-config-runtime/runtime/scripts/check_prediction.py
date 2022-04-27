from datetime import datetime

from loguru import logger
import numpy as np
import pandas as pd
import typer
from pathlib import Path

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
prediction_path = Path("/codeexecution/prediction.csv")
submission_format_path = Path("/codeexecution/data/partial_submission_format.csv")
columns = [
    "airport",
    "timestamp",
    "lookahead",
    "config",
]


def main(prediction_time: datetime):
    """Checks the predictions for a single timepoint to make sure the expected columns are present,
    all of the timestamps are equal to the expected timestamp, and the probabiilties for
    configurations at a single airport, timestamp, and lookahead sum to 1.
    """
    assert (
        prediction_path.exists()
    ), f"Submission did not generate prediction for {prediction_time}."

    prediction = pd.read_csv(prediction_path)
    submission_format = pd.read_csv(submission_format_path)

    if not submission_format.columns.equals(prediction.columns):
        raise ValueError(
            f"""Prediction columns ({", ".join(prediction.columns)}) do not match expected """
            f"""columns ({", ".join(submission_format.columns)})."""
        )

    if not submission_format[columns].equals(prediction[columns]):
        raise ValueError("Prediction index does not match submission format index.")

    assert (
        prediction.timestamp == prediction_time.strftime(DATETIME_FORMAT)
    ).all(), f"Prediction timestamps should all equal {prediction_time.strftime(DATETIME_FORMAT)}."

    airport_config_probability_sums = prediction.groupby(
        ["airport", "timestamp", "lookahead"]
    ).active.sum()
    if not np.allclose(airport_config_probability_sums, 1):
        logger.error(prediction)
        logger.error(airport_config_probability_sums)
        raise ValueError(
            "Probabilities for configurations at an airport, timestamp, lookahead do not sum to 1."
        )


if __name__ == "__main__":
    typer.run(main)
