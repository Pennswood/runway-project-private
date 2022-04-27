from pathlib import Path

from loguru import logger
import pandas as pd
import typer

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
WARM_START_PERIOD = "2D"
REPO_ROOT = Path(__file__).parents[1]

app = typer.Typer(add_completion=False)


def create_development_labels_and_submission_format(
    input_labels_path: Path, output_directory: Path
):
    """Creates development labels and submission format by selecting the final day from the open
    arena train labels.
    """
    logger.info("Creating development labels and submission format.")
    labels = pd.read_csv(input_labels_path, parse_dates=["timestamp"])
    end_time = labels.timestamp.max()
    start_time = (end_time - pd.Timedelta(days=1)).floor("1D")
    output_path = output_directory / "test_labels.csv"
    labels = labels.loc[
        (start_time <= labels.timestamp) & (labels.timestamp <= end_time)
    ].reset_index(drop=True)
    logger.debug(
        "Subsetting training labels to {:,} time points from {}-{} and saving to {}.",
        len(labels),
        start_time,
        end_time,
        output_path,
    )
    labels.to_csv(output_path, date_format=DATETIME_FORMAT, index=False)

    airport_uniform_probabilities = 1 / labels.groupby(
        "airport"
    ).config.nunique().rename("active")
    submission_format = (
        labels.drop(columns=["active"])
        .merge(airport_uniform_probabilities, on="airport")
        .reset_index(drop=True)
    )

    assert labels[["airport", "timestamp", "lookahead", "config"]].equals(
        submission_format[["airport", "timestamp", "lookahead", "config"]]
    ), "Submission format and label indices do not match."

    output_path = output_directory / "submission_format.csv"
    logger.debug(
        "Creating submission format with uniform probabilities and saving to {}.",
        output_path,
    )
    submission_format.to_csv(output_path, date_format=DATETIME_FORMAT, index=False)


def create_development_features(input_feature_directory: Path, output_directory: Path):
    """Creates development features by selecting the final day from the open arena features plus a
    warm start buffer of two days prior.
    """
    submission_format = pd.read_csv(
        output_directory / "submission_format.csv",
        parse_dates=["timestamp"],
        nrows=1,
    )
    feature_start = (
        submission_format.iloc[0].timestamp - pd.Timedelta(WARM_START_PERIOD)
    ).floor("1D")

    logger.info(f"Subsetting prescreened features to time points after {feature_start}")
    airport_directories = sorted(input_feature_directory.glob("k*"))
    if len(airport_directories) == 0:
        raise ValueError(
            f"No features detected in {input_feature_directory}. Does that directory contain one "
            "directory of features per airport? You may need to extract the tar archive if you "
            "downloaded the features from the competition data download page."
        )
    for airport_directory in airport_directories:
        airport = airport_directory.name
        logger.debug(f"Processing {airport}")
        for path in sorted(airport_directory.glob("*.csv.bz2")):
            output_path = output_directory / airport / path.name
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df.loc[df.timestamp >= feature_start].to_csv(
                output_path, date_format=DATETIME_FORMAT, index=False
            )

            assert pd.read_csv(path, nrows=1).columns.equals(
                pd.read_csv(output_path, nrows=1).columns
            ), "Original feature columns and prescreened subset columns do not match "
            f"({airport}, {path.stem})."


@app.command()
def main(
    input_feature_directory: Path = typer.Argument(
        REPO_ROOT / "data",
        help="Directory containing the training features, one directory of features per airport, "
        "e.g., data/katl, data/clt, data/den, etc.",
    ),
    input_labels_path: Path = typer.Argument(
        REPO_ROOT / "data" / "open_train_labels.csv.bz2",
        help="Path to the training labels.",
    ),
    output_directory: Path = typer.Option(
        REPO_ROOT / "runtime" / "data",
        help="Directory where the development dataset will be saved.",
    ),
):
    """Creates a data subset that matches the format of the full evaluation data intended for
    developing your submission.
    """
    output_directory.mkdir(exist_ok=True, parents=True)
    create_development_labels_and_submission_format(input_labels_path, output_directory)
    create_development_features(input_feature_directory, output_directory)


if __name__ == "__main__":
    app()
