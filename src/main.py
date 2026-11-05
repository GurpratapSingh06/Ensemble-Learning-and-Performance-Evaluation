"""
Practical Assignment 5: Ensemble Learning and Performance Evaluation
Email spam classification using the UCI Spambase dataset.

Run:
    pip install -r requirements.txt
    python src/main.py
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/spambase/spambase.data"
)


def load_dataset():
    """Download/load Spambase and separate features from binary target."""
    data = pd.read_csv(DATA_URL, header=None)
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].astype(int)
    X.columns = [f"feature_{i}" for i in range(1, 58)]
    return X, y


def build_models():
    """Create four base classifiers and three ensemble classifiers."""
    base_models = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", probability=True, random_state=42
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": GaussianNB()
    }

    stacking_estimators = [
        ("svm", base_models["SVM"]),
        ("decision_tree", base_models["Decision Tree"]),
        ("knn", base_models["KNN"]),
        ("naive_bayes", base_models["Naive Bayes"])
    ]
    stacking_model = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=42
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1
    )

    weak_learner = DecisionTreeClassifier(
        max_depth=1, random_state=42
    )
    adaboost_model = AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )

    voting_estimators = [
        ("svm", base_models["SVM"]),
        ("decision_tree", base_models["Decision Tree"]),
        ("knn", base_models["KNN"]),
        ("naive_bayes", base_models["Naive Bayes"])
    ]
    voting_model = VotingClassifier(
        estimators=voting_estimators,
        voting="hard",
        n_jobs=-1
    )

    all_models = {
        **base_models,
        "Stacking": stacking_model,
        "AdaBoost": adaboost_model,
        "Voting": voting_model
    }
    return all_models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train each model once and calculate classification metrics."""
    results = []
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Hard Voting does not expose predict_proba; log loss is N/A.
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            loss = log_loss(y_test, y_prob, labels=[0, 1])
        else:
            loss = np.nan

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Log Loss": loss,
            "Training Time": training_time
        })

        print(
            f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}, "
            f"Training time={training_time:.4f}s"
        )

    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, predictions


def save_outputs(results_df, predictions, y_test):
    """Save metrics, graphs, confusion matrices, and reports."""
    results_df.to_csv(OUTPUT_DIR / "ensemble_performance_comparison.csv")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    ax = results_df[metrics].plot(
        kind="bar", figsize=(13, 7), ylim=(0, 1.05), width=0.8
    )
    ax.set_title("Performance Comparison: Base Models vs Ensemble Models")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Performance Score")
    plt.xticks(rotation=30)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_comparison.png", dpi=300)
    plt.close()

    results_df["Log Loss"].dropna().plot(
        kind="bar", figsize=(10, 5)
    )
    plt.title("Log Loss Comparison (Lower is Better)")
    plt.xlabel("Classification Models")
    plt.ylabel("Log Loss")
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "log_loss_comparison.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (name, y_pred) in zip(axes.flatten(), predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Spam", "Spam"],
            yticklabels=["Non-Spam", "Spam"], ax=ax
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
    axes.flatten()[-1].axis("off")
    plt.suptitle("Confusion Matrices: Base and Ensemble Classifiers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300)
    plt.close()

    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        for name, y_pred in predictions.items():
            f.write(f"\n{'=' * 60}\nCLASSIFICATION REPORT: {name}\n{'=' * 60}\n")
            f.write(classification_report(
                y_test, y_pred,
                target_names=["Non-Spam", "Spam"],
                zero_division=0
            ))

    print(f"\nAll result files saved in: {OUTPUT_DIR}")


def main():
    print("Loading UCI Spambase dataset...")
    X, y = load_dataset()
    print(f"Records: {len(X)} | Features: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts().rename(index={0: "Non-Spam", 1: "Spam"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = build_models()
    results_df, predictions = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )

    print("\nFINAL PERFORMANCE COMPARISON")
    print(results_df.round(4).to_string())

    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        best = results_df[metric].idxmax()
        print(f"Highest {metric}: {best} ({results_df.loc[best, metric]:.4f})")

    best_loss = results_df["Log Loss"].idxmin()
    print(
        f"Lowest Log Loss: {best_loss} "
        f"({results_df.loc[best_loss, 'Log Loss']:.4f})"
    )

    save_outputs(results_df, predictions, y_test)


if __name__ == "__main__":
    main()
