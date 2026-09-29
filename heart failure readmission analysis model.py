# ================================================================
# HEART FAILURE 30-DAY READMISSION PREDICTION
# ================================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ----------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Readmitted_30_Days"

# Change this only if your CSV has a different filename.
CSV_FILE = "dataset_12000_records.csv"

# Folder where every required graph/table will be saved.
OUTPUT_DIR = Path("heart_failure_project_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    # Helpful fallback: automatically find a CSV in the current folder.
    csv_files = list(Path(".").glob("*.csv"))
    if len(csv_files) == 1:
        CSV_FILE = str(csv_files[0])
        print(f"CSV_FILE not found by the specified name.")
        print(f"Using the only CSV found: {CSV_FILE}")
    else:
        raise FileNotFoundError(
            f"Could not find '{CSV_FILE}'. Put the CSV in the same folder "
            f"as this Python file or change CSV_FILE at the top."
        )

df = pd.read_csv(CSV_FILE)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum().sum())
print("\nDuplicate rows:", df.duplicated().sum())

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the CSV.")

# ----------------------------------------------------------------
# 3. TARGET DISTRIBUTION GRAPH
# ----------------------------------------------------------------
class_counts = df[TARGET].value_counts().sort_index()
class_percent = class_counts / len(df) * 100

plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Not Readmitted (0)", "Readmitted (1)"],
    class_counts.values
)
plt.title("Target Class Distribution")
plt.xlabel("Readmission Status")
plt.ylabel("Number of Patients")
plt.grid(axis="y", alpha=0.25)

for bar, count, pct in zip(bars, class_counts.values, class_percent.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------------------------------------------
# 4. REMOVE IDENTIFIER / DEFINE FEATURES
# ----------------------------------------------------------------
drop_columns = [TARGET]

if "Patient_ID" in df.columns:
    drop_columns.append("Patient_ID")

X = df.drop(columns=drop_columns)
y = df[TARGET]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ----------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ----------------------------------------------------------------
# 6. PREPROCESSING PIPELINES
# ----------------------------------------------------------------
scaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

# ----------------------------------------------------------------
# 7. MODELS + HYPERPARAMETER SEARCH
# ----------------------------------------------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": (
        Pipeline([
            ("pre", scaled_preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE
            ))
        ]),
        {
            "model__C": [0.01, 0.1, 1, 10]
        }
    ),

    "KNN": (
        Pipeline([
            ("pre", scaled_preprocessor),
            ("model", KNeighborsClassifier())
        ]),
        {
            "model__n_neighbors": [3, 5, 7, 11, 15]
        }
    ),

    "Decision Tree": (
        Pipeline([
            ("pre", tree_preprocessor),
            ("model", DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_leaf": [1, 5, 10]
        }
    )
}

# ----------------------------------------------------------------
# 8. TRAIN, TUNE AND EVALUATE
# ----------------------------------------------------------------
trained_models = {}
results = []

for name, (pipeline, param_grid) in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    model = search.best_estimator_
    trained_models[name] = model

    print("Best parameters:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    # ---- TRAINING SET ----
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]

    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred, zero_division=0)
    train_recall = recall_score(y_train, train_pred, zero_division=0)
    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    train_auc = roc_auc_score(y_train, train_prob)

    # ---- TEST SET ----
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, test_prob)

    cm = confusion_matrix(y_test, test_pred)

    print("\nTRAINING PERFORMANCE")
    print(f"Accuracy : {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall   : {train_recall:.4f}")
    print(f"F1-score : {train_f1:.4f}")
    print(f"ROC-AUC  : {train_auc:.4f}")

    print("\nTEST PERFORMANCE")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1-score : {test_f1:.4f}")
    print(f"ROC-AUC  : {test_auc:.4f}")

    print("\nTEST CONFUSION MATRIX")
    print(cm)

    results.append({
        "Model": name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Precision": train_precision,
        "Test Precision": test_precision,
        "Train Recall": train_recall,
        "Test Recall": test_recall,
        "Train F1": train_f1,
        "Test F1": test_f1,
        "Train ROC-AUC": train_auc,
        "Test ROC-AUC": test_auc,
        "Accuracy Gap": train_accuracy - test_accuracy,
        "ROC-AUC Gap": train_auc - test_auc,
        "CV ROC-AUC": search.best_score_,
        "Best Parameters": str(search.best_params_)
    })

results_df = pd.DataFrame(results)

# ----------------------------------------------------------------
# 9. CONFUSION MATRIX GRAPHS — ONE FOR EACH MODEL
# ----------------------------------------------------------------
for i, (name, model) in enumerate(trained_models.items(), start=2):

    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} — Test Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    threshold = cm.max() / 2.0

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                f"{cm[row, col]:,}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold"
            )

    plt.tight_layout()

    filename = {
        "Logistic Regression": "02_logistic_regression_confusion_matrix.png",
        "KNN": "03_knn_confusion_matrix.png",
        "Decision Tree": "04_decision_tree_confusion_matrix.png"
    }[name]

    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# ----------------------------------------------------------------
# 10. METRIC COMPARISON GRAPH
# ----------------------------------------------------------------
metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
model_names = results_df["Model"].tolist()

x = np.arange(len(model_names))
width = 0.15

plt.figure(figsize=(11, 6))

for j, metric in enumerate(metrics):
    values = results_df[metric].values
    bars = plt.bar(
        x + (j - 2) * width,
        values,
        width,
        label=metric.replace("Test ", "")
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.xticks(x, model_names)
plt.ylim(0, 1.08)
plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Test Performance Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_metric_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------------------------------------------
# 11. ROC CURVE — ALL THREE MODELS
# ----------------------------------------------------------------
plt.figure(figsize=(8, 6))

for name, model in trained_models.items():

    probability = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probability)
    auc = roc_auc_score(y_test, probability)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Test Set")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_roc_curves.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------------------------------------------
# 12. TRAINING VS TEST ACCURACY
# ----------------------------------------------------------------
x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(9, 6))

train_values = results_df["Train Accuracy"].values
test_values = results_df["Test Accuracy"].values

bars1 = plt.bar(x - width / 2, train_values, width, label="Training Accuracy")
bars2 = plt.bar(x + width / 2, test_values, width, label="Testing Accuracy")

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, model_names)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Training vs Testing Accuracy — Overfitting Analysis")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_train_test_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------------------------------------------
# 13. OPTIONAL BUT RECOMMENDED: TRAINING VS TEST ROC-AUC
# ----------------------------------------------------------------
plt.figure(figsize=(9, 6))

train_auc_values = results_df["Train ROC-AUC"].values
test_auc_values = results_df["Test ROC-AUC"].values

bars1 = plt.bar(x - width / 2, train_auc_values, width, label="Training ROC-AUC")
bars2 = plt.bar(x + width / 2, test_auc_values, width, label="Testing ROC-AUC")

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, model_names)
plt.ylim(0, 1.05)
plt.ylabel("ROC-AUC")
plt.xlabel("Model")
plt.title("Training vs Testing ROC-AUC — Generalization Analysis")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_train_test_roc_auc.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------------------------------------------
# 14. OVERFITTING / UNDERFITTING DIAGNOSIS
# ----------------------------------------------------------------
print("\n" + "=" * 70)
print("OVERFITTING / UNDERFITTING ANALYSIS")
print("=" * 70)

for _, row in results_df.iterrows():

    acc_gap = abs(row["Train Accuracy"] - row["Test Accuracy"])
    auc_gap = abs(row["Train ROC-AUC"] - row["Test ROC-AUC"])

    if acc_gap < 0.02 and auc_gap < 0.02:
        diagnosis = "GOOD GENERALIZATION"
    elif acc_gap < 0.03 and auc_gap < 0.04:
        diagnosis = "MILD OVERFITTING"
    else:
        diagnosis = "MODERATE/STRONG OVERFITTING"

    print(f"\n{row['Model']}")
    print(f"  Accuracy gap : {acc_gap:.4f}")
    print(f"  ROC-AUC gap  : {auc_gap:.4f}")
    print(f"  Diagnosis    : {diagnosis}")

# ----------------------------------------------------------------
# 15. SAVE FINAL COMPARISON TABLE
# ----------------------------------------------------------------
results_df.to_csv(
    OUTPUT_DIR / "model_comparison_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

display_columns = [
    "Model",
    "Train Accuracy",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1",
    "Test ROC-AUC",
    "Accuracy Gap",
    "ROC-AUC Gap"
]

print(results_df[display_columns].round(4).to_string(index=False))

best_model_row = results_df.loc[results_df["Test ROC-AUC"].idxmax()]

print("\nBEST MODEL")
print("Model:", best_model_row["Model"])
print(f"Test Accuracy: {best_model_row['Test Accuracy']:.4f}")
print(f"Test Precision: {best_model_row['Test Precision']:.4f}")
print(f"Test Recall: {best_model_row['Test Recall']:.4f}")
print(f"Test F1: {best_model_row['Test F1']:.4f}")
print(f"Test ROC-AUC: {best_model_row['Test ROC-AUC']:.4f}")

print("\nAll required graphs have been displayed and saved in:")
print(OUTPUT_DIR.resolve())

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nAnalysis completed successfully.")
