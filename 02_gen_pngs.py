import os 
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = BASE_DIR / "data" / "test.parquet"
MODEL_OUTPUT_PATH = BASE_DIR / "artifacts" / "churn_model_logreg_tuned.pkl"

# Load data
df = pd.read_parquet(TEST_DATA_PATH)
X_test = df.drop("Churn", axis=1)
y_test = df["Churn"]

# Load model and preprocessor
with open(MODEL_OUTPUT_PATH, "rb") as f:
    pipe = pickle.load(f)

# Predict
y_pred = pipe.predict(X_test)

# Add prediction to DataFrame
df["prediction"] = y_pred

# Create output directory if not exists
os.makedirs("output", exist_ok=True)

# Generate At-Risk Customers chart
at_risk_count = (df["prediction"] == 1).sum()

plt.figure(figsize=(4, 4))
plt.text(0.5, 0.5, f"{at_risk_count:,}\nAt-Risk\nCustomers", 
         ha='center', va='center', fontsize=18)
plt.axis("off")
plt.savefig("output/at_risk_customers.png", bbox_inches="tight")
plt.close()

# Generate Tenure vs Churn Prediction Chart
tenure_churn = df[df["prediction"] == 1].groupby("tenure").size()

plt.figure(figsize=(10, 6))
tenure_churn.plot(kind="bar")
plt.xlabel("Tenure")
plt.ylabel("Predicted # of Churning Customers")
plt.title("Telco Churn - Predictions")
plt.tight_layout()
plt.savefig("output/tenure_churn_predictions.png")
plt.close()

# Generate Customer Breakdown Pie Chart
churn_counts = df["prediction"].value_counts()
labels = ["Not Churning", "Churning"]
colors = ["blue", "red"]

plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Customer Breakdown - Telco Churn")
plt.savefig("output/customer_breakdown_pie.png")
plt.close()