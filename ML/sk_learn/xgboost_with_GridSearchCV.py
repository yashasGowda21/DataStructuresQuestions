# -----------------------------------
# Setup and Data Generation
# -----------------------------------

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --- 1. Generate Synthetic Data (for demonstration) ---
np.random.seed(42)

num_samples = 1000

data = {
    'MonthlyCharges': np.random.normal(50, 20, num_samples),
    'TotalCharges': np.random.normal(1500, 500, num_samples),
    'Tenure': np.random.randint(1, 72, num_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'HasFiberOptic': np.random.choice([0, 1], num_samples),
    'CallsMade': np.random.randint(0, 300, num_samples),
    'DataUsageGB': np.random.uniform(0, 100, num_samples),
    'Churn': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 30% churn
}

df = pd.DataFrame(data)

# Introduce some missing values for demonstration of XGBoost's handling
for col in ['MonthlyCharges', 'TotalCharges']:
    missing_indices = np.random.choice(df.index, size=int(0.05 * num_samples), replace=False)
    df.loc[missing_indices, col] = np.nan

print("Sample Data Head:")
print(df.head())
print("\nMissing values before XGBoost:")
print(df.isnull().sum())

# --- 2. Preprocessing (for categorical features, XGBoost handles missing numeric) ---
# Encode categorical features
for col in ['Contract', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")




# ----------------------------------------------
# Basic XGBoost Model Training
# ----------------------------------------------


# --- 3. Basic XGBoost Model Training ---
print("\n--- Training Basic XGBoost Model ---")

# For classification, use XGBClassifier
model = xgb.XGBClassifier(
    objective='binary:logistic', # For binary classification
    eval_metric='logloss',       # Metric for evaluation during training
    use_label_encoder=False,     # Suppress warning for older label encoder behavior
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of churn

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# ------------------------------------------------------------------------------------
# Hyperparameter Tuning with GridSearchCV and Early Stopping
# ------------------------------------------------------------------------------------

# --- 4. Hyperparameter Tuning with GridSearchCV and Early Stopping ---
print("\n--- Hyperparameter Tuning with GridSearchCV and Early Stopping ---")

# Define a smaller parameter grid for demonstration (can be expanded)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.7, 0.9]
}

# XGBoost with Early Stopping
# The eval_set and early_stopping_rounds arguments are crucial here.
# It monitors the performance on the validation set (X_test, y_test)
# and stops if no improvement is seen for a specified number of rounds.
xgb_tuned_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=xgb_tuned_model,
    param_grid=param_grid,
    scoring='roc_auc', # Optimize for ROC AUC
    cv=3,              # 3-fold cross-validation
    verbose=1,
    n_jobs=-1          # Use all available CPU cores
)

# Fit with early stopping
# Note: Early stopping in GridSearchCV with XGBoost requires passing eval_set
# to the .fit() method of the estimator directly if you want it to be part
# of the GridSearch's internal fit. However, a simpler way for demonstration
# is to let GridSearchCV fit the model and then apply early stopping in a separate
# step or pass eval_set to the GridSearchCV.fit() itself if the estimator supports it.
# For simplicity in this example, we'll demonstrate early stopping in a direct fit.
# For GridSearchCV, it's typically about finding the best params, and then
# fitting the final model with early stopping.

# Let's run a simple fit with early stopping to illustrate the concept.
# In a real scenario, you'd integrate this with GridSearchCV's fitting logic if possible
# or re-fit the best model with early stopping after GridSearchCV.

print("\nFitting a single model with early stopping for demonstration:")
xgb_early_stop = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=1000, # Start with a large number, let early stopping decide
    learning_rate=0.05
)

xgb_early_stop.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   early_stopping_rounds=50, # Stop if validation score doesn't improve for 50 rounds
                   verbose=False) # Set to True to see boosting rounds output

print(f"Best iteration (n_estimators found by early stopping): {xgb_early_stop.best_iteration}")

# Now, use GridSearchCV to find best parameters (without early stopping directly inside GridSearchCV's fit,
# as it's more complex to combine directly for every fold; usually you find parameters and then apply early stopping for the final fit)
print("\nRunning GridSearchCV to find optimal hyperparameters (this might take a moment):")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best ROC AUC score from GridSearchCV: {grid_search.best_score_:.4f}")

# Train final model with best parameters and early stopping (best practice)
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    **grid_search.best_params_, # Unpack best parameters
    n_estimators=1000 # Use a large number, early stopping will adjust
)

final_model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False)

final_y_pred = final_model.predict(X_test)
final_y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print(f"\nFinal Model Accuracy (with tuned params & early stopping): {accuracy_score(y_test, final_y_pred):.4f}")
print(f"Final Model ROC AUC (with tuned params & early stopping): {roc_auc_score(y_test, final_y_pred_proba):.4f}")
print("\nFinal Model Classification Report:")
print(classification_report(y_test, final_y_pred))