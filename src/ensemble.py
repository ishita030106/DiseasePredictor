# evaluate_models.py

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from ensemble import (
    soft_voting_ensemble,
    weighted_voting_ensemble,
    stacking_ensemble
)

from train_rf import train_random_forest
from train_xgb import train_xgboost
from train_lr import train_logistic_regression
from train_svm import train_svm


# ============================================================
# EVALUATION FUNCTION
# ============================================================


def evaluate_model(model_name, model, X_test, y_test):

    print(f"\n{'=' * 50}")
    print(f"EVALUATING : {model_name}")
    print(f"{'=' * 50}\n")

    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(X_test)

    # ========================================================
    # PROBABILITIES
    # ========================================================

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 0

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")

    print(confusion_matrix(y_test, y_pred))

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================


def evaluate_all_models(X_train, X_test, y_train, y_test):

    results = []

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_model = train_random_forest(X_train, y_train)

    rf_result = evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test
    )

    results.append(rf_result)

    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_model = train_xgboost(X_train, y_train)

    xgb_result = evaluate_model(
        "XGBoost",
        xgb_model,
        X_test,
        y_test
    )

    results.append(xgb_result)

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    lr_model = train_logistic_regression(X_train, y_train)

    lr_result = evaluate_model(
        "Logistic Regression",
        lr_model,
        X_test,
        y_test
    )

    results.append(lr_result)

    # ========================================================
    # SVM
    # ========================================================

    svm_model = train_svm(X_train, y_train)

    svm_result = evaluate_model(
        "SVM",
        svm_model,
        X_test,
        y_test
    )

    results.append(svm_result)

    # ========================================================
    # SOFT VOTING ENSEMBLE
    # ========================================================

    soft_model = soft_voting_ensemble(
        X_train,
        X_test,
        y_train,
        y_test
    )

    soft_result = evaluate_model(
        "Soft Voting Ensemble",
        soft_model,
        X_test,
        y_test
    )

    results.append(soft_result)

    # ========================================================
    # WEIGHTED VOTING ENSEMBLE
    # ========================================================

    weighted_model = weighted_voting_ensemble(
        X_train,
        X_test,
        y_train,
        y_test
    )

    weighted_result = evaluate_model(
        "Weighted Voting Ensemble",
        weighted_model,
        X_test,
        y_test
    )

    results.append(weighted_result)

    # ========================================================
    # STACKING ENSEMBLE
    # ========================================================

    stacking_model = stacking_ensemble(
        X_train,
        X_test,
        y_train,
        y_test
    )

    stacking_result = evaluate_model(
        "Stacking Ensemble",
        stacking_model,
        X_test,
        y_test
    )

    results.append(stacking_result)

    # ========================================================
    # CREATE RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(results)

    # ========================================================
    # SORT BY F1 SCORE
    # ========================================================

    results_df = results_df.sort_values(
        by='F1 Score',
        ascending=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(results_df)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        'model_comparison_results.csv',
        index=False
    )

    print("\nResults saved as model_comparison_results.csv")

    return results_df

