from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
import pickle

RANDOM_STATE = 333
BASE_DIR = Path(__file__).resolve().parent
TELCO_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "artifacts" / "churn_model_logreg_tuned.pkl"
TEST_DATA_PATH = BASE_DIR / "data" / "test.parquet"

# Load & clean
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
df = pd.read_csv(TELCO_PATH)

# Convert TotalCharges to numeric, drop rows where conversion failed
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# Map target to 0/1, drop ID column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop(columns=["customerID"], inplace=True)

# Split features/target
X = df.drop(columns="Churn")
y = df["Churn"]

# Identify columns by dtype
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Train/val/test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)

# Save to Parquet file
# Combine test features and target for monthly simulated data
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_parquet(TEST_DATA_PATH, index=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

# make pipeline
tuned_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", TunedThresholdClassifierCV(
        estimator=LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        scoring=make_scorer(fbeta_score, beta=2),
        cv=5,
        refit=True,
        random_state=RANDOM_STATE
    ))
])

tuned_pipe.fit(X_train, y_train)
y_pred = tuned_pipe.predict(X_val)

print(f"Optimal threshold (CV): {tuned_pipe['clf'].best_threshold_:.3f}")
print("F2 score on validation set:", fbeta_score(y_val, y_pred, beta=2))

# Re-fit on all data after validation
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])
tuned_pipe.fit(X_final, y_final)

# Save
with open(MODEL_OUTPUT_PATH, "wb") as f:
    pickle.dump(tuned_pipe, f)