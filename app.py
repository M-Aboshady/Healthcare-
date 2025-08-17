import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- FILE PATHS ---
file_path = "/kaggle/input/er-csv/synthetic_uae_er_insurance_5000_balanced_clean.csv"
MODEL_PATH = "er_claim_model.joblib"
ENCODERS_PATH = "er_encoders.joblib"
FEATURES_PATH = "er_model_features.joblib"

# --- MAPPING FOR TARGET VARIABLE ---
CLAIM_STATUS_MAP = {'Approved': 0, 'Rejected': 1, 'Might_be_Approved': 2}
CLASS_LABELS = ['Approved', 'Rejected', 'Might_be_Approved']


def load_preprocess_and_train_model(file_path):
    """
    Loads the CSV data, preprocesses it (encodes categorical features and target),
    trains a RandomForestClassifier, and saves the model, encoders, and feature list.
    """

    # Step 1 - Load Data
    df = pd.read_csv(file_path)
    st.write("📥 Data loaded:", df.shape)
    print("📥 Data loaded:", df.shape)

    # Step 2 - Define features
    numerical_features = ['Age', 'Systolic_BP', 'Diastolic_BP', 
                          'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    categorical_features = ['Gender', 'CPT_Code', 'ICD_Code', 
                            'Insurance_Company', 'Insurance_Plan']
    target = 'Claim_Status'
    all_features = numerical_features + categorical_features

    st.write("🔑 Features defined")
    print("🔑 Features defined")

    # Step 3 - Copy data for encoding
    df_encoded = df.copy()
    encoders = {}

    # Step 4 - Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        st.write(f"✅ Encoded {col}")
        print(f"✅ Encoded {col}")

    # Step 5 - Encode target
    y_encoded = df_encoded[target].map(CLAIM_STATUS_MAP)
    if y_encoded.isna().sum() > 0:
        raise ValueError("❌ Found NaN in Claim_Status mapping! Check categories.")

    y_encoded = y_encoded.astype(int)
    st.write("🎯 Target encoded, unique classes:", y_encoded.unique())
    print("🎯 Target encoded, unique classes:", y_encoded.unique())

    # Step 6 - Define X and y
    X = df_encoded[all_features]
    y = y_encoded
    st.write("📊 Feature matrix shape:", X.shape, " Target shape:", y.shape)
    print("📊 Feature matrix shape:", X.shape, " Target shape:", y.shape)

    # Step 7 - Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    st.write("🤖 Model trained successfully")
    print("🤖 Model trained successfully")

    # Step 8 - Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(all_features, FEATURES_PATH)
    st.success("✅ Model, encoders, and features saved")
    print("✅ Model, encoders, and features saved")

    return model, encoders, all_features, X, y


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    model, encoders, all_features, X, y = load_preprocess_and_train_model(file_path)
    print("🏁 Done. Model trained on", X.shape[0], "rows and", X.shape[1], "features.")
    

# --- SAFE ENCODER FUNCTION ---
def safe_transform(le, value):
    """Safely transform unseen values into 'unknown' if possible."""
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # Extend encoder with "Unknown"
        le_classes = list(le.classes_)
        if "Unknown" not in le_classes:
            le_classes.append("Unknown")
            le.classes_ = np.array(le_classes)
        return le.transform(["Unknown"])[0]

# --- PREDICTION FUNCTION ---
def predict_claim(input_data):
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    all_features = joblib.load(FEATURES_PATH)

    # Convert to dataframe
    df_input = pd.DataFrame([input_data])

    # Encode categorical inputs safely
    for col, le in encoders.items():
        df_input[col] = df_input[col].apply(lambda x: safe_transform(le, str(x)))

    X_input = df_input[all_features]
    prediction = model.predict(X_input)[0]
    return CLASS_LABELS[prediction]

# --- STREAMLIT APP ---
st.set_page_config(page_title="UAE ER Claim Predictor", layout="wide")
st.title("🏥 UAE ER Claim Approval Prediction App")

# Layout split into 3 columns
col1, col2, col3 = st.columns([1,1,1])

# --- SECTION 1: VITAL SIGNS ---
with col1:
    st.header("🫀 Vital Signs")
    Age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=30.0, step=0.1)
    Systolic_BP = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=250.0, value=120.0, step=0.1)
    Diastolic_BP = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=150.0, value=80.0, step=0.1)
    Heart_Rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
    Temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
    Respiratory_Rate = st.number_input("Respiratory Rate (/min)", min_value=5.0, max_value=60.0, value=18.0, step=0.1)

# --- SECTION 2: DEMOGRAPHICS & INSURANCE ---
with col2:
    st.header("👤 Demographics & Insurance")
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Insurance_Company = st.text_input("Insurance Company", "Daman")
    Insurance_Plan = st.text_input("Insurance Plan", "Basic")
    ICD_Code = st.text_input("ICD Code", "S72.001A")

# --- SECTION 3: CPT & PREDICTION ---
with col3:
    st.header("📝 CPT & Prediction")
    CPT_Code = st.text_input("CPT Code", "99283")

    if st.button("🔮 Predict Claim Status"):
        input_data = {
            "Age": Age,
            "Gender": Gender,
            "Systolic_BP": Systolic_BP,
            "Diastolic_BP": Diastolic_BP,
            "Heart_Rate": Heart_Rate,
            "Temperature": Temperature,
            "Respiratory_Rate": Respiratory_Rate,
            "CPT_Code": CPT_Code,
            "ICD_Code": ICD_Code,
            "Insurance_Company": Insurance_Company,
            "Insurance_Plan": Insurance_Plan
        }
        result = predict_claim(input_data)
        st.success(f"📌 Predicted Claim Status: **{result}**")


