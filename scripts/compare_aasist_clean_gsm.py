import pandas as pd

clean = pd.read_csv(
    "results/aasist_evaluation/2021_predictions.csv"
)

gsm = pd.read_csv(
    "results/aasist_gsm_2021_predictions.csv"
)

merged = clean[
    ["filename", "actual_label", "deepfake_score"]
].merge(
    gsm[
        ["filename", "actual_label", "deepfake_score"]
    ],
    on=["filename", "actual_label"],
    suffixes=("_clean", "_gsm"),
)

merged["score_change"] = (
    merged["deepfake_score_gsm"]
    - merged["deepfake_score_clean"]
)

print("Matched samples:", len(merged))
print()

print("Score change: GSM - Clean")
print(
    merged.groupby("actual_label")["score_change"].describe()
)

print()
print("Mean deepfake score:")
print(
    merged.groupby("actual_label")[
        ["deepfake_score_clean", "deepfake_score_gsm"]
    ].mean()
)
