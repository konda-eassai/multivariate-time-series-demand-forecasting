import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    layout="wide"
)

st.title("Multivariate Demand Forecasting Dashboard")

st.markdown("""
This dashboard demonstrates **ML and Deep Learning based demand forecasting**
using historical retail sales data.

**Models available:**
- Random Forest (ML)
- LSTM (Deep Learning)
- GRU (Deep Learning)
""")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "C:\\Multivariate_TimeSeries_Forecasting_CP2\\data\\raw\\features.csv",
        parse_dates=["Date"],
        index_col="Date"
    )
    return df


@st.cache_resource
def load_models():
    rf = joblib.load("models/ml/random_forest.pkl")
    lstm = load_model("models/dl/lstm_model.h5", compile=False)
    gru = load_model("models/dl/gru_model.h5", compile=False)

    return rf, lstm, gru

df = load_data()
rf_model, lstm_model, gru_model = load_models()

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Forecasting Model",
    ["Random Forest", "LSTM", "GRU"]
)


TARGET = "Weekly_Sales"
X = df.drop(columns=[TARGET])
y = df[TARGET]

split_idx = int(len(X) * 0.8)
split_date = X.index.sort_values()[split_idx]


X_test = X[X.index > split_date]
y_test = y[y.index > split_date]


if model_choice == "Random Forest":
    y_pred = rf_model.predict(X_test)

else:
    # Load DL sequences
    X_seq = np.load("data/processed/X_lstm.npy")
    y_seq = np.load("data/processed/y_lstm.npy")

    split_idx = int(0.8 * len(X_seq))
    X_test_seq = X_seq[split_idx:]
    y_test = y_seq[split_idx:]

    if model_choice == "LSTM":
        y_pred = lstm_model.predict(X_test_seq).flatten()
    else:
        y_pred = gru_model.predict(X_test_seq).flatten()


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

col1, col2 = st.columns(2)

col1.metric("RMSE", f"{rmse:,.2f}")
col2.metric("MAPE", f"{mape:.2%}")


st.subheader("Actual vs Forecast (Sample)")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test[:200], label="Actual")
ax.plot(y_pred[:200], label="Forecast", alpha=0.7)
ax.legend()

st.pyplot(fig)


risk_df = pd.DataFrame({
    "Actual": y_test,
    "Forecast": y_pred
})

risk_df["Error_Percent"] = (
    (risk_df["Forecast"] - risk_df["Actual"]) / risk_df["Actual"]
)

risk_df["Risk_Type"] = "Normal"
risk_df.loc[risk_df["Error_Percent"] < -0.20, "Risk_Type"] = "Stock-Out Risk"
risk_df.loc[risk_df["Error_Percent"] > 0.20, "Risk_Type"] = "Overstock Risk"


st.subheader("Inventory Risk Distribution")

risk_counts = risk_df["Risk_Type"].value_counts()

fig, ax = plt.subplots(figsize=(5,3))
bars = ax.bar(risk_counts.index, risk_counts.values)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha="center",
        va="bottom"
    )

st.pyplot(fig)


st.markdown("""
### Key Insight
Deep learning models (LSTM / GRU) capture temporal demand patterns
more effectively than classical ML models, enabling proactive
inventory risk management.
""")
