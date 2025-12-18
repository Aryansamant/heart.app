from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import optuna
import mlflow


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "heart.db"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
RESULTS_CSV = METRICS_DIR / "experiment_results.csv"

MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)


# -------------------------
# Config
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Keep trials reasonable for laptops; increase if you want better tuning
OPTUNA_TRIALS = 25

# For tuning objective (CV F1)
CV_FOLDS = 5


@dataclass
class Experiment:
    exp_id: int
    model_name: str
    use_pca: bool
    use_optuna: bool


def load_df_from_db() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT p.*, l.target
            FROM patients p
            JOIN labels l USING(patient_id)
            """,
            conn,
        )
    return df


def make_base_pipeline(model, use_pca: bool) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        # Keep enough components; can be tuned too, but we’ll keep fixed for rubric simplicity
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def get_model(model_name: str, params: dict | None = None):
    params = params or {}

    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)

    if model_name == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)

    if model_name == "svm":
        # probability=True is helpful later for API/UI probability output
        return SVC(probability=True, random_state=RANDOM_STATE, **params)

    if model_name == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)

    raise ValueError(f"Unknown model_name: {model_name}")


def optuna_search_space(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        }

    if model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    if model_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    if model_name == "gb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    raise ValueError(f"Unknown model_name: {model_name}")


def tune_with_optuna(model_name: str, X_train, y_train, use_pca: bool) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = optuna_search_space(trial, model_name)
        model = get_model(model_name, params)
        pipe = make_base_pipeline(model, use_pca)

        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    return study.best_params


def run_one_experiment(exp: Experiment, X_train, X_test, y_train, y_test) -> dict:
    # Choose params
    best_params = {}
    if exp.use_optuna:
        best_params = tune_with_optuna(exp.model_name, X_train, y_train, exp.use_pca)

    # Train final model
    model = get_model(exp.model_name, best_params)
    pipe = make_base_pipeline(model, exp.use_pca)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    # Save model
    model_filename = f"exp_{exp.exp_id:02d}_{exp.model_name}_pca{int(exp.use_pca)}_optuna{int(exp.use_optuna)}.pkl"
    model_path = MODELS_DIR / model_filename
    joblib.dump(pipe, model_path)

    # Return record
    return {
        "exp_id": exp.exp_id,
        "model_name": exp.model_name,
        "use_pca": exp.use_pca,
        "use_optuna": exp.use_optuna,
        "f1_score": float(f1),
        "best_params": json.dumps(best_params),
        "model_path": str(model_path),
    }


def main():
    # Load from DB (rubric requirement)
    df = load_df_from_db()

    X = df.drop(columns=["patient_id", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Define the 16 experiments (4 models × 4 conditions)
    model_list = ["logreg", "rf", "svm", "gb"]
    conditions = [
        (False, False),  # no PCA, no tuning
        (False, True),   # no PCA, tuning
        (True, False),   # PCA, no tuning
        (True, True),    # PCA, tuning
    ]

    experiments: list[Experiment] = []
    exp_id = 1
    for m in model_list:
        for use_pca, use_optuna in conditions:
            experiments.append(Experiment(exp_id, m, use_pca, use_optuna))
            exp_id += 1

    # MLflow run (Dagshub-ready)
    mlflow.set_experiment("heart_disease_16_experiments")

    results = []
    for exp in experiments:
        with mlflow.start_run(run_name=f"exp_{exp.exp_id:02d}_{exp.model_name}_pca{int(exp.use_pca)}_optuna{int(exp.use_optuna)}"):
            mlflow.log_param("model_name", exp.model_name)
            mlflow.log_param("use_pca", exp.use_pca)
            mlflow.log_param("use_optuna", exp.use_optuna)
            mlflow.log_param("optuna_trials", OPTUNA_TRIALS if exp.use_optuna else 0)

            record = run_one_experiment(exp, X_train, X_test, y_train, y_test)

            mlflow.log_metric("f1_score", record["f1_score"])
            mlflow.log_param("best_params", record["best_params"])

            # Log model file as artifact
            mlflow.log_artifact(record["model_path"])

            results.append(record)
            print(f"✅ Exp {exp.exp_id:02d} | {exp.model_name} | PCA={exp.use_pca} | Optuna={exp.use_optuna} | F1={record['f1_score']:.4f}")

    results_df = pd.DataFrame(results).sort_values("exp_id")
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n====================")
    print("✅ All 16 experiments complete.")
    print(f"✅ Results saved to: {RESULTS_CSV}")
    print(f"✅ Models saved in: {MODELS_DIR}")
    print("====================\n")


if __name__ == "__main__":
    main()
