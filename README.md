# Practical Assignment 5 – Ensemble Learning and Performance Evaluation

## Problem statement
Implement and evaluate ensemble-learning techniques for email spam classification using the UCI Spambase dataset. Compare Stacking, AdaBoost and majority Voting with individual classification models.

## Student / practical details
- **Student:** Gurpratap Singh Sandhu
- **PRN:** 202401040323
- **Branch:** Computer Engineering
- **Division:** B
- **Batch:** B3 (AIML-A1)
- **Roll No.:** 145
- **Practical No.:** 5
- **Faculty:** Dr. Khushal Khairnar

## Algorithms and models

### Base classifiers
1. Support Vector Machine (SVM) with RBF kernel and feature standardization.
2. Decision Tree (`max_depth=10`).
3. K-Nearest Neighbours (KNN, `n_neighbors=5`) with feature standardization.
4. Gaussian Naïve Bayes.

### Ensemble classifiers
- **Stacking:** SVM, Decision Tree, KNN and Gaussian Naïve Bayes are base estimators. Logistic Regression is the meta-classifier. Five-fold cross-validation and `predict_proba` are used.
- **AdaBoost:** Decision Tree stump (`max_depth=1`), 100 estimators and learning rate 1.0.
- **Voting:** Hard voting (majority voting) across the four base classifiers.

## Dataset
- **Name:** UCI Spambase
- **Source:** [UCI Machine Learning Repository – Spambase](https://archive.ics.uci.edu/dataset/94/spambase)
- **Records:** 4,601
- **Features:** 57
- **Target:** 0 = non-spam, 1 = spam

The dataset is downloaded automatically from the UCI archive by the program. It is not bundled in this repository.

## Project structure

```text
aiml_practical_5_ensemble_learning/
├── notebooks/
│   └── Ensemble_Learning_and_Performance_Evaluation.ipynb
├── src/
│   └── main.py
├── outputs/
│   └── (generated after running the program)
├── tests/
│   └── test_setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and execution

### Option 1: Google Colab
1. Open `notebooks/Ensemble_Learning_and_Performance_Evaluation.ipynb` in Google Colab.
2. Run **Runtime → Run all**.
3. The first cell installs the required libraries.
4. The notebook downloads the Spambase dataset from UCI.

### Option 2: Run locally
Python 3.10 or later is recommended.

```bash
# Clone the repository or extract the ZIP
cd aiml_practical_5_ensemble_learning

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the experiment
python src/main.py
```

Internet access is required on first run to download the dataset.

## Evaluation metrics
- **Accuracy:** Fraction of all predictions classified correctly.
- **Precision:** Fraction of predicted spam emails that are actually spam.
- **Recall:** Fraction of actual spam emails correctly detected.
- **F1-score:** Harmonic mean of precision and recall.
- **Log loss:** Evaluates predicted class probabilities; lower is better.
- **Training time:** Time taken to fit each model.

**Note:** Hard Voting does not provide class-probability estimates in this implementation, so its log loss is recorded as N/A.

## Results from the submitted notebook run

| Model | Accuracy | Precision | Recall | F1-score | Log loss |
|---|---:|---:|---:|---:|---:|
| SVM | 92.73% | 92.77% | 88.43% | 90.55% | 0.2310 |
| Decision Tree | 90.88% | 89.74% | 86.78% | 88.24% | 1.3062 |
| KNN | 90.77% | 88.61% | 87.88% | 88.24% | 0.9458 |
| Naïve Bayes | 83.39% | 71.78% | 95.32% | 81.89% | 4.2251 |
| Stacking | **93.49%** | 92.20% | 91.18% | **91.69%** | **0.1785** |
| AdaBoost | 92.73% | 92.53% | 88.71% | 90.58% | 0.5274 |
| Voting | 93.38% | **93.64%** | 89.26% | 91.40% | N/A |

These are the recorded results from the submitted notebook execution. Training times may vary by machine and runtime.

## Expected output files
After running `src/main.py`, the `outputs/` directory contains:
- `ensemble_performance_comparison.csv`
- `performance_comparison.png`
- `log_loss_comparison.png`
- `confusion_matrices.png`
- `classification_reports.txt`

## Test cases
| Test | Input / action | Expected result |
|---|---|---|
| Dataset loading | Download and load Spambase | 4,601 rows and 57 features |
| Train-test split | Stratified 80:20 split, random state 42 | 3,680 training and 921 testing records |
| Model evaluation | Train four base and three ensemble models | Metrics and predictions for all seven models |
| Output generation | Complete the evaluation | CSV, graphs, confusion matrices and reports saved under `outputs/` |

## Conclusion
The practical compares individual classifiers with Stacking, AdaBoost and majority Voting. In the submitted notebook run, Stacking achieved the highest accuracy and F1-score, Voting achieved the highest precision, and Gaussian Naïve Bayes achieved the highest recall. The most appropriate model depends on the metric and application requirements.

## Future scope
- Evaluate soft voting and compare it with hard voting.
- Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
- Use repeated stratified cross-validation for more robust estimates.
- Test on additional email datasets and monitor false-positive rates.
- Explore feature selection and class-imbalance handling.

## License / dataset note
This repository contains student-authored code and notebook material. The Spambase dataset remains subject to the terms and attribution requirements of its source repository; it is downloaded at runtime rather than redistributed here.
