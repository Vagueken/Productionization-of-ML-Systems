from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# -------------------
# Step Functions
# -------------------

def extract_data():
    df = pd.read_csv("/Users/home/Downloads/travel-ml-capstone/Data/flights.csv")
    df.to_csv("/tmp/flights_raw.csv", index=False)

def preprocess_data():
    df = pd.read_csv("/tmp/flights_raw.csv")
    df = df.dropna()

    # Drop the raw 'date' column since it's not numeric
    if "date" in df.columns:
        df = df.drop("date", axis=1)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["from","to","flightType","agency"])

    df.to_csv("/tmp/flights_processed.csv", index=False)


def train_model():
    df = pd.read_csv("/tmp/flights_processed.csv")
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("/tmp/flight_price_model.pkl", "wb") as f:
        pickle.dump(model, f)

def validate_model():
    import joblib
    from sklearn.metrics import mean_squared_error
    import numpy as np

    df = pd.read_csv("/tmp/flights_processed.csv")
    X = df.drop("price", axis=1)
    y = df["price"]

    model = pickle.load(open("/tmp/flight_price_model.pkl","rb"))
    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"Model RMSE: {rmse}")

# -------------------
# DAG Definition
# -------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    "flight_price_pipeline",
    default_args=default_args,
    description="Flight Price Regression DAG",
    schedule="@daily",  # ✅ use 'schedule' instead of 'schedule_interval'
    start_date=datetime(2025, 9, 3),
    catchup=False,
) as dag:

    task_extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    # DAG dependencies
    task_extract >> task_preprocess >> task_train >> task_validate
