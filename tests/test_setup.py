"""Lightweight checks for project setup and model definitions."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import build_models  # noqa: E402


def test_seven_models_are_defined():
    models = build_models()
    assert len(models) == 7
    assert set(models) == {
        "SVM", "Decision Tree", "KNN", "Naive Bayes",
        "Stacking", "AdaBoost", "Voting"
    }


def test_voting_uses_hard_voting():
    voting = build_models()["Voting"]
    assert voting.voting == "hard"


def test_adaboost_uses_decision_stump():
    adaboost = build_models()["AdaBoost"]
    assert adaboost.n_estimators == 100
    assert adaboost.estimator.max_depth == 1
