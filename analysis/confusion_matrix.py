from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "heart.db"
OUTPUT_PATH = BASE_DIR / "analysis" / "confusion_matrix.png"

def main():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("""
            SELECT p.*, l.target
            FROM patients p
            JOIN labels l USING(patient_id)
        """, conn)

    X = df.drop(columns=["patient_id", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print(f"✅ Baseline F1-score: {f1:.4f}")

    ConfusionMatrixDisplay.from_estimator(
        pipeline,
        X_test,
        y_test,
        cmap="Blues"
    )

    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()

    print(f"✅ Confusion matrix saved at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
