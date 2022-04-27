from datetime import datetime
from pathlib import Path
from time import perf_counter
import shutil

from loguru import logger
import pandas as pd
from tqdm.contrib.concurrent import process_map
import typer

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

test_feature_directory = Path("/data")
extract_directory = Path("/extracts")


def create_feature_extracts(args):
    """Creates time-censored extracts of a feature for a sequence of prediction times.

    Args:
        args (tuple of pathlib.Path, datetime): Since `process_map` needs a function that takes a
            single argument, accept a tuple where the values are `feature_path`, the path a the
            feature CSV containing the full time range, and `prediction_times`, a sequence of
            prediction times to generate extracts for.
    """
    feature_path, prediction_times = args
    # Faster to not parse dates and just read in timestamps as strings
    df = pd.read_csv(feature_path)
    for prediction_time_str in prediction_times.strftime(DATETIME_FORMAT):
        output_relative_path = feature_path.relative_to(test_feature_directory)

        if feature_path.name == "submission_format.csv":
            valid_indices = df.timestamp == prediction_time_str
            output_relative_path = (
                output_relative_path.parent / "partial_submission_format.csv"
            )
        else:
            valid_indices = df.timestamp < prediction_time_str

        extract_path = extract_directory / prediction_time_str / output_relative_path
        extract_path.parent.mkdir(exist_ok=True, parents=True)

        df.loc[valid_indices].to_csv(extract_path, index=False)


def main(
    prediction_time: datetime,
    batch_period: str = "24H",
    batch_freq: str = "1H",
):
    prediction_time_str = prediction_time.strftime(DATETIME_FORMAT)
    # Check if directory exists for prediction time
    if (
        extract_directory.exists()
        and (extract_directory / prediction_time_str).exists()
    ):
        logger.debug("Features for {} exist in extract cache.", prediction_time_str)
        return

    start_time = perf_counter()

    batch_start = prediction_time
    batch_end = prediction_time + pd.Timedelta(batch_period) - pd.Timedelta(batch_freq)
    logger.info(f"Generating data extracts from {batch_start} to {batch_end}.")
    logger.debug("Deleting extract cache.")
    if extract_directory.exists():
        shutil.rmtree(extract_directory)
        extract_directory.mkdir(exist_ok=True, parents=True)

    feature_paths = (
        feature_path for feature_path in test_feature_directory.rglob("*.csv")
    )
    prediction_times = pd.date_range(batch_start, batch_end, freq=batch_freq)

    process_map(
        create_feature_extracts,
        [(feature_path, prediction_times) for feature_path in sorted(feature_paths)],
        chunksize=8,
    )

    total_time = perf_counter() - start_time
    logger.info(
        "Created extracts for {} prediction times in {:.2f} seconds "
        "({:.2f} seconds per prediction time).",
        len(prediction_times),
        total_time,
        total_time / len(prediction_times),
    )

    logger.debug("Updating permissions")
    for path in extract_directory.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)


if __name__ == "__main__":
    typer.run(main)
