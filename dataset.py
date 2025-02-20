import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(int(time.time()))
n_samples = 1000

# Define feature ranges
credit_score_range = (300, 850)
income_range = (10000, 500000)
overdue_range = (0, 15)
age_range = (18, 70)
employment_status_prob = [0.2, 0.3, 0.5]  

# Helper functions
def determine_credit_limit(credit_score, income):
    """Determine credit limit based on credit score and income."""
    if credit_score < 580:
        base_limit = 1000
        max_limit = min(3000, income // 50)
    elif credit_score < 670:
        base_limit = 2000
        max_limit = min(10000, income // 40)
    elif credit_score < 740:
        base_limit = 5000
        max_limit = min(20000, income // 30)
    elif credit_score < 800:
        base_limit = 10000
        max_limit = min(30000, income // 20)
    else:
        base_limit = 20000
        max_limit = min(50000, income // 10)
    
    max_limit = max(base_limit, max_limit)
    if max_limit <= base_limit:
        max_limit = base_limit + 100  
    
    return np.random.choice(np.arange(base_limit, max_limit + 1, 100))

def determine_line_of_credit_duration(age):
    """Determine the duration of a line of credit based on the user's age."""
    earliest_start_age = 14
    max_start_age = min(age, 25)
    start_age = np.random.choice(np.arange(earliest_start_age, max_start_age + 1))
    return age - start_age

# Scaling functions
def scale_credit_score(score):
    return 1 - ((score - 300) / (850 - 300))

def scale_overdue_payments(overdue):
    return np.log1p(overdue) / np.log1p(15)

def scale_balance_ratio(balance, credit_limit):
    return min(balance / credit_limit, 1)

def scale_employment_and_income(status, income, age):
    if status == 0:
        return 0.7 if age >= 60 and income < 30000 else 1.0
    elif status == 1:
        return 0.6 if income < 50000 else 0.4
    return 0.3 if income < 50000 else 0.1

# Generate dataset
data = []
for _ in range(n_samples):
    credit_score = np.random.randint(*credit_score_range)
    income = np.random.randint(*income_range)
    employment_status = np.random.choice([0, 1, 2], p=employment_status_prob)
    age = np.random.randint(*age_range)
    duration_of_credit = determine_line_of_credit_duration(age)
    
    credit_limit = determine_credit_limit(credit_score, income)
    highest_balance = np.random.randint(int(0.5 * credit_limit), credit_limit + 1)
    current_balance = np.random.randint(0, highest_balance + 1)
    overdue_payments = np.random.randint(*overdue_range)
    
    # Scale features
    credit_score_contrib = scale_credit_score(credit_score)
    overdue_payments_contrib = scale_overdue_payments(overdue_payments)
    highest_balance_contrib = scale_balance_ratio(highest_balance, credit_limit)
    current_balance_contrib = scale_balance_ratio(current_balance, credit_limit)
    employment_income_contrib = scale_employment_and_income(employment_status, income, age)
    
    # Compute risk score
    risk_score = (
        0.25 * credit_score_contrib +
        0.2 * overdue_payments_contrib +
        0.15 * highest_balance_contrib +
        0.15 * current_balance_contrib +
        0.25 * employment_income_contrib
    )
    risk_score = min(risk_score, 1)
    
    # Append row
    data.append({
        "credit_score": credit_score,
        "credit_limit": credit_limit,
        "total_overdue_payments": overdue_payments,
        "highest_balance": highest_balance,
        "current_balance": current_balance,
        "income": income,
        "age": age,
        "employment_status": employment_status,
        "duration_of_credit": duration_of_credit,
        "risk_score": risk_score
    })


df = pd.DataFrame(data)

# Save dataset
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "realistic_credit_risk_data.csv")
if os.path.exists(data_path):
    choice = input("Do you want to overwrite the existing dataset? (yes/no): ").strip().lower()
    if choice == "yes":
        df.to_csv(data_path, index=False)
    else:
        df.to_csv(data_path.replace(".csv", f"_{int(time.time())}.csv"), index=False)
else:
    df.to_csv(data_path, index=False)


models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
}

# Train-test split
X = df.drop(columns=["risk_score"])
y = df["risk_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate models
model_path = os.path.join(script_dir, "loan_risk_model.pkl")
performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    previous_r2 = joblib.load(model_path).score(X_test, y_test) if os.path.exists(model_path) else None
    performance_change = r2 - previous_r2 if previous_r2 is not None else 0
    print(f"{name}: MSE={mse:.4f}, R²={r2:.4f} model performance {performance_change:+.4f}")
    performance[name] = r2


for name, model in models.items():
    model_filename = os.path.join(script_dir, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_filename)
    print(f"{name} model saved at: {model_filename}")

best_model = max(models, key=lambda k: performance[k])
joblib.dump(models[best_model], model_path)
print(f"🏆 Best model saved at: {model_path}")
