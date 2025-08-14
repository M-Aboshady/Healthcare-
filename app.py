
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
# --- FILE PATHS ---
# 🚨 IMPORTANT: Update these paths to your file locations.
EXCEL_FILE_PATH = "synthetic_uae_er_insurance_5000_balanced_clean.xlsx"
MODEL_PATH = "er_claim_model.joblib"
ENCODERS_PATH = "er_encoders.joblib"
FEATURES_PATH = "er_model_features.joblib"

def load_data(file_path):
    """Loads and cleans the Excel data for training."""
    try:
        df = pd.read_excel(file_path)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the path is correct.")
        return None

def train_model(df):
    """Trains a Random Forest Classifier and saves the model, encoders, and feature list."""
    if df is None or df.empty:
        st.warning("Cannot train model: data is empty or not loaded.")
        return None

    # Define all features and the target variable from the prompt
    features = ['Age', 'Gender', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Respiratory_Rate', 'CPT_Code', 'ICD_Code', 'Insurance_Company', 'Insurance_Plan']
    target = 'Claim_Status'

    df_cleaned = df.dropna(subset=features + [target])

    encoders = {}
    for col in ['Gender', 'CPT_Code', 'ICD_Code', 'Insurance_Company', 'Insurance_Plan']:
        if col in df_cleaned.columns:
            le = LabelEncoder()
            df_cleaned[f'{col}_encoded'] = le.fit_transform(df_cleaned[col])
            encoders[col] = le
            features.append(f'{col}_encoded')
            features.remove(col) # Use encoded features for the model

    X = df_cleaned[features]
    y = df_cleaned[target]
    
    # Mapping the target variable
    y_encoded = y.map({'Approved': 0, 'Rejected': 1, 'Might Be Approved': 2})

    # Assuming 'y_encoded' is your target variable
    import numpy as np

# Check for NaN values
    if np.isnan(y_encoded).any():
        print("y_encoded contains NaN values.")
        # You can also see where they are
        nan_indices = np.where(np.isnan(y_encoded))
        print(f"NaN values found at indices: {nan_indices}")

# Check for infinite values
    if np.isinf(y_encoded).any():
        print("y_encoded contains infinite values.")
        

    model = RandomForestClassifier(n_estimators=100, random_state=42)


    df_combined = pd.DataFrame(X, columns=features)
    df_combined['target'] = y_encoded

    # Drop rows with NaN values from the combined DataFrame
    df_cleaned = df_combined.dropna()

    # Separate X and y again
    X_cleaned = df_cleaned.drop('target', axis=1)
    y_cleaned = df_cleaned['target']


    model.fit(X_cleaned, y_cleaned)
    

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(features, FEATURES_PATH)
    st.success("✅ Machine Learning model and encoders trained and saved successfully!")
    return model, encoders, features

def predict_from_user_input(model, encoders, features, user_input):
    """
    Predicts the claim status for a single user input by preprocessing the data entry.
    """
    # Create a DataFrame from the single user input
    input_df = pd.DataFrame([user_input])

    
    # Preprocess categorical features using the saved encoders
    for col, encoder in encoders.items():
        # Handle new, unseen categories gracefully
        if user_input[col] not in encoder.classes_:
            return "Might Be Approved", "Reason: Unseen category in historical data. Manual review required."
        input_df[f'{col}_encoded'] = encoder.transform([user_input[col]])

    # Reorder columns to match the trained model's feature order
    input_df = input_df[features]

    # Make the prediction
    prediction_proba = model.predict_proba(input_df)[0]
    predicted_class_idx = model.predict(input_df)[0]
    
    # Map back to readable labels
    class_labels = ['Approved', 'Rejected', 'Might Be Approved']
    prediction = class_labels[predicted_class_idx]
    confidence = prediction_proba[predicted_class_idx]

    return prediction, f"Confidence: {confidence:.2f}"

# --- STREAMLIT UI ---
st.set_page_config(page_title="ER Claim Pre-Approval", layout="wide")
st.title("🚑 ER Claim Pre-Approval Assistant")
st.markdown("---")

# Load data and train/load model
df_claims = load_data(EXCEL_FILE_PATH)
model, encoders, features = None, None, None

if df_claims is not None and not df_claims.empty:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH) and os.path.exists(FEATURES_PATH):
        st.info("Loading existing model, encoders, and features...")
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        features = joblib.load(FEATURES_PATH)
    else:
        st.info("Model not found. Training a new model now...")
        model, encoders, features = train_model(df_claims)

if model is not None and encoders is not None:
    # --- Input Sections ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1️⃣ Patient Demographics")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col2:
        st.subheader("2️⃣ Vital Signs")
        st.markdown("Enter the patient's vital signs.")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=250, value=80)
        bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=250, value=120)
        bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=80)
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
        respiration_rate = st.number_input("Respiratory_Rate (breaths/min)", min_value=0, max_value=60, value=18)
    
    # CPT and ICD codes section
    st.markdown("---")
    st.subheader("3️⃣ Claim Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        cpt_code = st.text_input("CPT Procedure Code", "99285")
    with col4:
        icd_code = st.text_input("ICD Diagnosis Code", "R10.9")
    with col5:
        insurance_company = st.text_input("Insurance Company", "Daman")
        insurance_plan = st.text_input("Insurance Plan", "Basic")

    # Prediction button
    submitted = st.button("Get Prediction")

    if submitted:
        user_input = {
            'Age': age,
            'Gender': gender,
            'Systolic_BP': bp_systolic,
            'Diastolic_BP': bp_diastolic,
            'Heart_Rate': heart_rate,
            'Temperature': temperature,
            'Respiratory_Rate': respiration_rate,
            'CPT_Code': cpt_code,
            'ICD_Code': icd_code,
            'Insurance_Company': insurance_company,
            'Insurance_Plan': insurance_plan
        }
        
        st.markdown("---")
        st.header("4️⃣ Prediction Result")
        st.spinner("Predicting claim status...")
        prediction, reason = predict_from_user_input(model, encoders, features, user_input)
        
        if prediction == 'Approved':
            st.success(f"**Prediction:** {prediction} ✅")
            st.write(f"Reason: {reason}")
        elif prediction == 'Might Be Approved':
            st.warning(f"**Prediction:** {prediction} ⚠️")
            st.write(f"Reason: {reason}")
        else:
            st.error(f"**Prediction:** {prediction} ❌")
            st.write(f"Reason: {reason}")
else:
    st.info("Please ensure your historical data file is in place to train the model.")












