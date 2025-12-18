from pathlib import Path
import sqlite3
import pandas as pd

# Project root: ~/Desktop/HEART_DISEASE
BASE_DIR = Path(__file__).resolve().parents[1]

# heart.csv is directly inside the project folder
DATA_PATH = BASE_DIR / "heart.csv"

DB_PATH = BASE_DIR / "database" / "heart.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in heart.csv")

    # Ensure database folder exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create DB + tables
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_PATH.read_text())

        # Add patient_id for normalization
        df = df.copy()
        df.insert(0, "patient_id", range(1, len(df) + 1))

        feature_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]

        patients_df = df[["patient_id"] + feature_cols]
        labels_df = df[["patient_id", "target"]]

        patients_df.to_sql("patients", conn, if_exists="append", index=False)
        labels_df.to_sql("labels", conn, if_exists="append", index=False)

    print("✅ Database created successfully")
    print(f"📍 Dataset loaded from: {DATA_PATH}")
    print(f"📍 Database stored at: {DB_PATH}")
    print(f"📊 Rows inserted: {len(df)}")


if __name__ == "__main__":
    main()
