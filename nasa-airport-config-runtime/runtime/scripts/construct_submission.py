from pathlib import Path

import pandas as pd
import typer

prediction_directory = Path("/predictions")
submission_path = Path("/codeexecution/submission/submission.csv")


def main():
    submission = []
    for path in prediction_directory.glob("*.csv"):
        submission.append(pd.read_csv(path))

    submission = pd.concat(submission, ignore_index=True).sort_values(
        [
            "airport",
            "timestamp",
            "lookahead",
            "config",
        ]
    )
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    typer.run(main)
