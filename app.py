import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# Preprocessing + Training
# ==============================
@st.cache_resource  # cache model training so it runs only once
def train_model():
    # Load dataset
    Hepatitis = pd.read_csv("HepatitisCdataF.csv")
    Hepatitis = Hepatitis.iloc[:, 1:]  # remove first col (id)

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    numeric_df = Hepatitis.select_dtypes(include="float64")
    imputed_Hepatitis = imputer.fit_transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_Hepatitis, columns=numeric_df.columns)
    Hepatitis[numeric_df.columns] = imputed_df

    # Handle categorical encoding
    label_encoder = LabelEncoder()
    Hepatitis["Category"] = label_encoder.fit_transform(Hepatitis["Category"])
    if "Sex" in Hepatitis.columns:
        Hepatitis["Sex"] = label_encoder.fit_transform(Hepatitis["Sex"])

    # Normalize features
    scaler = MinMaxScaler()
    numerical_features = Hepatitis.select_dtypes(include=["float64", "int64"]).columns
    Hepatitis[numerical_features] = scaler.fit_transform(Hepatitis[numerical_features])

    # Split data
    X = Hepatitis.drop("Category", axis=1)
    y = Hepatitis["Category"]

    # Handle imbalance
    bsmote = BorderlineSMOTE(random_state=42)
    X_res, y_res = bsmote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = gb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model + encoder
    joblib.dump(gb_model, "hepatitis_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return gb_model, label_encoder, scaler, acc, X.columns


# ==============================
# Streamlit UI
# ==============================
st.title("🦠 Hepatitis C Virus Prediction App")
st.write("This app predicts **Hepatitis C categories** using ML.")

# Train or Load model
model, label_encoder, scaler, feature_names, acc, cols = None, None, None, None, None, None
try:
    model = joblib.load("hepatitis_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    cols = joblib.load("feature_names.pkl")
except:
    model, label_encoder, scaler, acc, cols = train_model()
    joblib.dump(cols, "feature_names.pkl")

st.success(f"✅ Model trained with accuracy: {round(acc*100,2)}%")

# Collect user input
st.subheader("Enter Patient Details")

user_input = {}
for col in cols:
    if col == "Sex":
        user_input[col] = st.selectbox("Sex", ["male", "female"])
    else:
        user_input[col] = st.number_input(col, value=0.0)

# Convert input into dataframe
input_df = pd.DataFrame([user_input])

# Encode categorical fields
if "Sex" in input_df.columns:
    input_df["Sex"] = label_encoder.transform(input_df["Sex"])

# Scale numerical values
numeric_cols = input_df.select_dtypes(include=["float64", "int64"]).columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict
if st.button("Predict Hepatitis Category"):
    prediction = model.predict(input_df)[0]
    category = label_encoder.inverse_transform([prediction])[0]
    st.subheader(f"🧾 Prediction: {category}")
