# Setting up tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split # ML Library
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib #To save trained python object, Flask API will load it later
import os
from db_config import get_database_client
import json
os.makedirs('backend/models',exist_ok=True)

# Ingesting Data
db=get_database_client()
collection=db['fused_outbreak_data']
cursor = collection.find({},{'_id':0})
data=pd.DataFrame(list(cursor))
if data.empty:
    print('Database is empty. Execute pipeline.py first.')
    exit(1)

# Creating the answer key for supervised learning
numeric_cols = [
    'Avg_Temperature_2m',
    'Avg_Relative_Humidity_2m',
    'Search_Trend_Score',
    'Rainfall',
    'Cases_Last_Week',
    'Rainfall_Lag_1',
    'Temp_Humidity_Index',
    'Rainfall_Change',
    'Temperature_Change',
    'Humidity_Change',
    'Search_Trend_Momentum',
    'Reported_Cases'
]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
def assign_risk(cases):
    if cases > 150: return 2
    if cases > 50: return 1
    return 0
data['Risk_Level']=data['Reported_Cases'].apply(assign_risk)

# Splitting Inputs and Outputs
X = data[
[
    'Avg_Temperature_2m',
    'Avg_Relative_Humidity_2m',
    'Search_Trend_Score',
    'Rainfall',
    'Cases_Last_Week',
    'Rainfall_Lag_1',
    'Temp_Humidity_Index',
    'Rainfall_Change',
    'Temperature_Change',
    'Humidity_Change',
    'Search_Trend_Momentum'
]
] # This is Feature, Frontend will send to backend
y=data['Risk_Level'] # This is Target, answer which we want the model to predict

# Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)
#random_state=42 is to make sure data is shuffled the same way every time the script is run

# Training the brain
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
et_model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

gb_model = GradientBoostingClassifier(
    random_state=42
)
#n_estimators to define how many t]decision trees at a time
#max_depth to allow each decision tree to ask 10 y/n questions before making decision
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

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Confusion Matrix")
    print(cm)

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
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"\n{name}")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Low','Medium','High'],
        zero_division = 0
    ))

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
best_model_name = comparison_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

print(f"\nBest Model: {best_model_name}")

best_model = voting_model
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.named_estimators_['rf'].feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(feature_importance)

plt.figure(figsize=(10,5))

plt.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
