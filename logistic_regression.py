# CIE-2 Unit 2: Logistic Regression
# Run this file from the folder containing student_pass.csv.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

df = pd.read_csv("student_pass.csv")
features = ["Hours_Studied", "Attendance_Percent", "Previous_Score"]
X, y = df[features], df["Pass"]

# Stratified split preserves class proportions in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

# Majority-class baseline
majority_class = y_train.mode()[0]
baseline_pred = [majority_class] * len(y_test)
print("Majority-class baseline accuracy:", accuracy_score(y_test, baseline_pred))

# Evaluate different probability thresholds
for threshold in [0.3, 0.5, 0.7]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    print(f"\n--- Threshold = {threshold} ---")
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall/Sensitivity:", recall_score(y_test, predictions, zero_division=0))
    print("Specificity:", specificity)
    print("F1-score:", f1_score(y_test, predictions, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

# Example recommendation: choose threshold based on the cost of errors.
# If missing a student who will pass is costly, prioritize recall (lower threshold).
# If false pass predictions are costly, prioritize precision/specificity (higher threshold).

new_student = pd.DataFrame([{
    "Hours_Studied": 7, "Attendance_Percent": 80, "Previous_Score": 65
}])
new_probability = model.predict_proba(new_student)[0, 1]
print("\nNew student's probability of passing:", new_probability)
print("Predicted class at threshold 0.5:", int(new_probability >= 0.5))
