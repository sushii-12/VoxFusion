import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PATH = "results/aasist_gsm_2021_calibration.csv"

p = pd.read_csv(PATH)
y = (p["actual_label"] == "spoof").astype(int)

print("Threshold  Accuracy  Precision  Recall  F1      FAR     FRR")
print("-" * 65)

for t in [i / 100 for i in range(50, 100)]:
    predicted = (p["deepfake_score"] >= t).astype(int)

    accuracy = accuracy_score(y, predicted)
    precision = precision_score(y, predicted, zero_division=0)
    recall = recall_score(y, predicted, zero_division=0)
    f1 = f1_score(y, predicted, zero_division=0)

    false_positive = (
        (predicted == 1) & (p["actual_label"] == "bonafide")
    ).sum()

    false_negative = (
        (predicted == 0) & (p["actual_label"] == "spoof")
    ).sum()

    far = false_positive / 50
    frr = false_negative / 50

    print(
        f"{t:.2f}       "
        f"{accuracy:.3f}     "
        f"{precision:.3f}      "
        f"{recall:.3f}   "
        f"{f1:.3f}   "
        f"{far:.3f}   "
        f"{frr:.3f}"
    )