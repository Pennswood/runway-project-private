import pandas as pd

submission = pd.read_csv("/codeexecution/submission/submission.csv")
submission_format = pd.read_csv("/data/submission_format.csv")


def test_submission_conforms_to_format():
    assert submission[["airport", "timestamp", "lookahead", "config"]].equals(
        submission_format[["airport", "timestamp", "lookahead", "config"]]
    ), "Submission does not have the expected index."
