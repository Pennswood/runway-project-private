import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import typer

REPO_ROOT = Path(__file__).parents[1]

app = typer.Typer(add_completion=False)


@app.command()
def main(
    predictions_path: Path = typer.Option(REPO_ROOT / "submission" / "submission.csv"),
    labels_path: Path = typer.Option(
        REPO_ROOT / "runtime" / "data" / "test_labels.csv"
    ),
):
    """Computes the mean-aggregated log loss for a set of predictions given some test labels."""
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)
    airport_scores = {
        airport: log_loss(
            labels.loc[labels.airport == airport].active,
            predictions.loc[predictions.airport == airport].active,
            eps=1e-16,
        )
        for airport in sorted(labels.airport.unique())
    }
    logger.info("Airport scores:\n{}", json.dumps(airport_scores, indent=2))
    score = np.mean(list(airport_scores.values()))
    logger.success(f"Score: {score}")


if __name__ == "__main__":
    app()
