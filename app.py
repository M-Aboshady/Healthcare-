import streamlit as st

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_curve
)
from lightgbm import LGBMClassifier

# --- FILE PATHS ---
file_path = "synthetic_claims_uae_capitalized_3.csv"
MODEL_PATH = "er_claim_model_lgbm.joblib"
ENCODERS_PATH = "er_encoders_lgbm.joblib"
FEATURES_PATH = "er_model_features_lgbm.joblib"
VECTORIZER_PATH = "er_notes_vectorizer_lgbm.joblib"
THRESHOLD_PATH = "er_best_threshold_lgbm.joblib"

# --- MAPPING FOR TARGET VARIABLE ---
CLAIM_STATUS_MAP = {'Approved': 0, 'Rejected': 1}
CLASS_LABELS = ['Approved', 'Rejected']

def load_preprocess_and_train_model(file_path):
    """
    Loads, preprocesses, and trains a LightGBM model on the ER claims data.
    This includes handling structured and text data, and tuning for F1-score.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the path is correct.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check your CSV file format.")
        return None, None, None, None, None, None

    st.write("📥 Data loaded:", df.shape)
    print("📥 Data loaded:", df.shape)

    # Step 1 - Define features and target
    numerical_features = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    categorical_features = ['CPT_Code', 'ICD_Code', 'Insurance_Company', 'Insurance_Plan']
    text_feature = 'Clinical_Notes'
    target = 'Claim_Status'
    all_features = numerical_features + categorical_features + [text_feature]

    # Step 2 - Encode categorical features
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Step 3 - Encode target
    y = df_encoded[target].map(CLAIM_STATUS_MAP).astype(int)

    # Step 4 - Encode text (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=500)
    X_text_full = vectorizer.fit_transform(df_encoded[text_feature].astype(str))

    # Step 5 - Structured features matrix
    X_struct_full = df_encoded[numerical_features + categorical_features].values

    # Step 6 - Train/test split (stratified)
    Xs_train, Xs_test, Xt_train, Xt_test, y_train, y_test = train_test_split(
        X_struct_full,
        X_text_full,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 7 - Combine structured + text (train and test sets)
    X_train_combined = hstack([csr_matrix(Xs_train), Xt_train], format="csr")
    X_test_combined = hstack([csr_matrix(Xs_test), Xt_test], format="csr")

    # Step 8 - Train LightGBM model
    model = LGBMClassifier(
        n_estimators=200,
        class_weight="balanced", # to handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_combined, y_train)
    print("🤖 LightGBM model trained successfully")

    # Step 9 - Evaluate and save artifacts
    y_pred = model.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("✅ Validation Accuracy:", acc)
    print(report)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(all_features, FEATURES_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    return model, encoders, all_features, vectorizer, (X_test_combined, y_test, y_pred, acc, report, cm)

# --- SAFE ENCODER FUNCTION ---
def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        le_classes = list(le.classes_)
        if "Unknown" not in le_classes:
            le_classes.append("Unknown")
            le.classes_ = np.array(le_classes)
        return le.transform(["Unknown"])[0]

# --- PREDICTION FUNCTION ---
def predict_claim(input_data, clinical_note):
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        all_features = joblib.load(FEATURES_PATH)
    except FileNotFoundError as e:
        st.error(f"Prediction artifacts not found. Please run the training script first. Error: {e}")
        return "Error", 0, 0
    except Exception as e:
        st.error(f"An error occurred while loading model artifacts. Error: {e}")
        return "Error", 0, 0

    numerical_features = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    categorical_features = ['CPT_Code', 'ICD_Code', 'Insurance_Company', 'Insurance_Plan']
    
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical safely
    for col in categorical_features:
        if col in encoders:
            le = encoders[col]
            df_input[col] = df_input[col].apply(lambda x: safe_transform(le, str(x)))
    
    X_structured = df_input[numerical_features + categorical_features]
    X_text = vectorizer.transform([clinical_note])
    
    X_input = hstack([X_structured.values, X_text])
    
    prediction = model.predict(X_input)[0]
    
    return CLASS_LABELS[prediction]

# --- STREAMLIT APP UI ---

    st.set_page_config(page_title="ER Claim Prediction App", layout="wide")
    st.title("🏥 Emergency Room Claim Status Prediction")
    st.markdown("Enter the patient's vitals and clinical notes to predict if their claim will be **Approved** or **Rejected**.")

    # Load the dataset once to get unique values for validation
    try:
        df = pd.read_csv(DATA_PATH)
        known_companies = set(df['Insurance_Company'].unique())
        known_plans = set(df['Insurance_Plan'].unique())
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please ensure it is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data file: {e}")
        st.stop()
    

    # Input fields for the user
    with st.form("claim_form"):
        st.header("Patient Vitals & Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            heart_rate = st.number_input("Heart Rate", min_value=0.0, value=75.0)
            respiratory_rate = st.number_input("Respiratory Rate", min_value=0.0, value=16.0)
        with col2:
            systolic_bp = st.number_input("Systolic BP", min_value=0.0, value=120.0)
            diastolic_bp = st.number_input("Diastolic BP", min_value=0.0, value=80.0)
            temperature = st.number_input("Temperature (°F)", min_value=0.0, value=98.6)
            cpt_code = st.text_input("CPT Code", "99213")
        
        st.header("Claim Details")
        col3, col4 = st.columns(2)
        with col3:
            insurance_company = st.text_input("Insurance Company", "Daman")
            icd_code = st.text_input("ICD Code", "I10")
        with col4:
            insurance_plan = st.text_input("Insurance Plan", "Basic")
        
        st.header("Clinical Notes")
        clinical_note = st.text_area("Clinical Notes", "Patient stable, normal findings")
        
        submitted = st.form_submit_button("Predict Claim Status")

    if submitted:
        # Check if the insurance company or plan is new
        is_new_insurance_company = insurance_company not in known_companies
        is_new_insurance_plan = insurance_plan not in known_plans

        if is_new_insurance_company or is_new_insurance_plan:
            st.warning("New insurance or new plan")

        input_data = {
            'Age': age,
            'Gender': gender,
            'Heart_Rate': heart_rate,
            'Respiratory_Rate': respiratory_rate,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'Temperature': temperature,
            'ICD_Code': icd_code,
            'CPT_Code': cpt_code,
            'Insurance_Company': insurance_company,
            'Insurance_Plan': insurance_plan,
        }
        
        # Call the prediction function
        prediction = predict_claim(input_data, clinical_note)
        
        # Display the result
        if prediction == "Error":
            st.warning("Prediction could not be completed. Please check the logs.")
        else:
            st.success(f"The model predicts the claim status is: **{prediction}**")



# -------------------- MAIN EXECUTION & FEATURE IMPORTANCE --------------------
if __name__ == "__main__":
    try:
        model, encoders, all_features, vectorizer, results = load_preprocess_and_train_model(file_path)
        X_test, y_test, y_pred, acc, report, cm = results
        print("🏁 Done. Model trained and evaluated.")

    
        
        

        st.markdown("---")
        st.subheader("🤖 Feature Importance")
        
        # Get feature names from structured data and TF-IDF vectorizer
        structured_feature_names = ['Age_encoded', 'CPT_Code_encoded', 'ICD_Code_encoded', 'Insurance_Company_encoded', 'Insurance_Plan_encoded'] # Adjust to match your encoding scheme
        text_feature_names = vectorizer.get_feature_names_out().tolist()
        
        # Combine all feature names
        feature_names = structured_feature_names + text_feature_names
        
        # Map feature importance to names
        importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        
        # Sort by importance
        importances = importances.sort_values('importance', ascending=False).head(20) # Show top 20
        
        
    except Exception as e:
        st.error(f"An error occurred during training or evaluation: {e}")
        print(f"An error occurred during training or evaluation: {e}")

