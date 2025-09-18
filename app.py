import pandas as pd
import joblib
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

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

    # Encode categorical variables
    category_encoder = LabelEncoder()
    Hepatitis["Category"] = category_encoder.fit_transform(Hepatitis["Category"])

    sex_encoder = None
    if "Sex" in Hepatitis.columns:
        sex_encoder = LabelEncoder()
        Hepatitis["Sex"] = sex_encoder.fit_transform(Hepatitis["Sex"])

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

    # Train model
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = gb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save everything
    joblib.dump(gb_model, "hepatitis_model.pkl")
    joblib.dump(category_encoder, "category_encoder.pkl")
    if sex_encoder:
        joblib.dump(sex_encoder, "sex_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns, "feature_names.pkl")

    return gb_model, category_encoder, sex_encoder, scaler, acc, X.columns


# ==============================
# Streamlit UI
# ==============================
st.title("🦠 Hepatitis C Virus Prediction App")
st.write("This app predicts **Hepatitis C categories** using ML.")

# Train or Load model
try:
    model = joblib.load("hepatitis_model.pkl")
    category_encoder = joblib.load("category_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    cols = joblib.load("feature_names.pkl")
    try:
        sex_encoder = joblib.load("sex_encoder.pkl")
    except:
        sex_encoder = None
    acc = None
except:
    model, category_encoder, sex_encoder, scaler, acc, cols = train_model()

if acc:
    st.success(f"✅ Model trained with accuracy: {round(acc*100,2)}%")
else:
    st.info("Model loaded from saved file.")

# Collect user input
st.subheader("Enter Patient Details")

user_input = {}
for col in cols:
    if col == "Sex":
        user_input[col] = st.selectbox("Sex", ["Male", "Female"])
    else:
        user_input[col] = st.number_input(col, value=0.0)

# Convert input into dataframe
input_df = pd.DataFrame([user_input])

# Encode Sex safely
if "Sex" in input_df.columns and sex_encoder is not None:
    input_df["Sex"] = sex_encoder.transform(input_df["Sex"])

# Scale numerical values
numeric_cols = input_df.select_dtypes(include=["float64", "int64"]).columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict
if st.button("Predict Hepatitis Category"):
    prediction = model.predict(input_df)[0]
    category = category_encoder.inverse_transform([prediction])[0]
    st.subheader(f"🧾 Prediction: {category}")
