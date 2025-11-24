import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, balanced_accuracy_score
)

def main(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    y = df["HeartDisease"]
    X = df.drop(columns=["HeartDisease"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = joblib.load(model_path)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("PR AUC:", average_precision_score(y_test, proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC curve")
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # PR curve
    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Precision-Recall curve")
    plt.savefig("reports/pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots to reports/")

if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path do pliku heart.csv")
    parser.add_argument("--model", required=True, help="Path do model.joblib")
    args = parser.parse_args()
    main(args.data, args.model)