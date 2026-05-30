from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import joblib
# Training Models

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)

et_model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

gb_model = GradientBoostingClassifier(
    random_state=42
)

voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('et', et_model)
    ],
    voting='soft'
)

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Extra Trees": et_model,
    "Gradient Boosting": gb_model,
    "Voting Ensemble": voting_model
}

results = []

best_accuracy = 0
best_model = None
best_model_name = ""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

for name, model in models.items():

    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

    print(f"\n{name}")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

comparison_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nMODEL COMPARISON")
print(comparison_df)

print(f"\nBEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")

# Save best model
joblib.dump(
    best_model,
    'backend/models/disease_predictor.pkl'
)

data = pd.read_csv("your_dataset.csv")
X = data['temperature',
    'humidity',
    'rainfall']



# Feature Importance
if hasattr(rf_model, "feature_importances_"):

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print(feature_importance)

    plt.figure(figsize=(10, 5))

    plt.bar(
        feature_importance['Feature'],
        feature_importance['Importance']
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()