from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    return rf_model

importance_df = pd.DataFrame({

    'Feature': X_train.columns,

    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print(importance_df)

plt.figure(figsize=(12, 8))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()