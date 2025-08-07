import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model and scaler
with open("notebooks/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("notebooks/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@st.cache_data
def load_data():
    df = pd.read_csv("data/WineQT.csv")

    # Drop non-numeric columns
    object_cols = df.select_dtypes(include="object").columns
    if len(object_cols) > 0:
        st.warning(f"Dropping non-numeric columns: {list(object_cols)}")
        df = df.drop(columns=object_cols)

    # Convert everything to standard float64
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce')).astype("float64")

    return df


df = load_data()

# Page config
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("🍷 Wine Quality Prediction App")
st.markdown("""
This application allows you to:
- Explore the wine quality dataset
- Visualize feature relationships
- Make real-time predictions using a trained ML model
- Evaluate model performance
""")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", [
    "Data Exploration",
    "Visualisations",
    "Model Prediction",
    "Model Performance"
])

# DATA EXPLORATION
if menu == "Data Exploration":
    st.header("🔍 Data Overview")

    st.write("*Dataset Shape:*", df.shape)
    st.write("*Data Types:*")
    st.dataframe(df.dtypes)

    st.write("*Sample Data:*")
    st.dataframe(df.head())

    st.subheader("🔧 Filter Data")
    filter_col = st.selectbox("Select a column to filter", df.select_dtypes(include=np.number).columns)
    min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
    user_range = st.slider(f"Select range for {filter_col}", min_val, max_val, (min_val, max_val))
    filtered_df = df[(df[filter_col] >= user_range[0]) & (df[filter_col] <= user_range[1])]
    st.write("Filtered Data:")
    st.dataframe(filtered_df)

# VISUALIZATION
elif menu == "Visualisations":
    st.header("📊 Visualisation Section")

    st.subheader("1. Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Alcohol vs Quality")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='quality', y='alcohol', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Volatile Acidity Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['volatile acidity'], kde=True, ax=ax3)
    st.pyplot(fig3)

# MODEL PREDICTION
elif menu == "Model Prediction":
    st.header("🔮 Predict Wine Quality")
    st.markdown("Enter the chemical properties of the wine below:")

    with st.form("wine_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.0)
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
            citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0)
            residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=15.0, value=1.9)

        with col2:
            chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0)
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=250.0, value=34.0)
            density = st.number_input("Density", min_value=0.9900, max_value=1.0100, value=0.9978)

        with col3:
            pH = st.number_input("pH", min_value=2.0, max_value=4.5, value=3.51)
            sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56)
            alcohol = st.number_input("Alcohol", min_value=5.0, max_value=15.0, value=9.4)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                pH, sulphates, alcohol]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled).max()

        st.success(f"🍷 Predicted Wine Quality: *{prediction}*")
        st.info(f"📊 Prediction Confidence: *{probability * 100:.2f}%*")

# MODEL PERFORMANCE
elif menu == "Model Performance":
    st.header("📈 Model Performance")

    from sklearn.model_selection import train_test_split

    # Drop columns not used during model training
    X = df.drop(["quality", "Id", "quality_label"], axis=1, errors="ignore")
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure column match with what scaler was fitted on
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"*Accuracy:* {acc:.2f}")

    st.write("*Classification Report:*")
    st.text(classification_report(y_test, y_pred))

    st.write("*Confusion Matrix:*")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)