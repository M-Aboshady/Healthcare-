import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np

# --- FILE PATHS ---
file_path = "synthetic_uae_er_insurance_5000_with_notes.csv"
MODEL_PATH = "er_claim_model_with_nlp.joblib"
ENCODERS_PATH = "er_encoders_with_nlp.joblib"
FEATURES_PATH = "er_model_features_with_nlp.joblib"
VECTORIZER_PATH = "er_notes_vectorizer.joblib"

# --- MAPPING FOR TARGET VARIABLE ---
CLAIM_STATUS_MAP = {'Approved': 0, 'Rejected': 1, 'Might_be_Approved': 2}
CLASS_LABELS = ['Approved', 'Rejected', 'Might_be_Approved']

def load_preprocess_and_train_model(file_path):
    # Step 1 - Load Data
    df = pd.read_csv(file_path)
    st.write("📥 Data loaded:", df.shape)

    # Step 2 - Define features
    numerical_features = ['Age', 'Systolic_BP', 'Diastolic_BP', 
                          'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    categorical_features = ['Gender', 'CPT_Code', 'ICD_Code', 
                            'Insurance_Company', 'Insurance_Plan']
    text_feature = 'Clinical_Notes'
    target = 'Claim_Status'
    all_features = numerical_features + categorical_features + [text_feature]

    # Step 3 - Encode categorical features
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Step 4 - Encode target
    y_encoded = df_encoded[target].map(CLAIM_STATUS_MAP).astype(int)

    # Step 5 - Encode text (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=500)  # limit features for speed
    X_text = vectorizer.fit_transform(df_encoded[text_feature].astype(str))

    # Step 6 - Structured features
    X_structured = df_encoded[numerical_features + categorical_features]

    # Step 7 - Combine structured + text
    from scipy.sparse import hstack
    X_combined = hstack([X_structured.values, X_text])

    # Step 8 - Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_combined, y_encoded)

    # Step 9 - Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(all_features, FEATURES_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, encoders, all_features, vectorizer

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
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    all_features = joblib.load(FEATURES_PATH)

    df_input = pd.DataFrame([input_data])

    # Encode categorical safely
    for col, le in encoders.items():
        df_input[col] = df_input[col].apply(lambda x: safe_transform(le, str(x)))

    X_structured = df_input.drop(columns=["Clinical_Notes"], errors="ignore")
    X_text = vectorizer.transform([clinical_note])

    X_input = hstack([X_structured.values, X_text])
    prediction = model.predict(X_input)[0]
    return CLASS_LABELS[prediction]

# --- STREAMLIT APP ---
st.set_page_config(page_title="UAE ER Claim Predictor with NLP", layout="wide")
st.title("🏥 UAE ER Claim Approval Prediction App (with Clinical Notes NLP)")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.header("🫀 Vital Signs")
    Age = st.number_input("Age", 0.0, 120.0, 30.0)
    Systolic_BP = st.number_input("Systolic BP", 50.0, 250.0, 120.0)
    Diastolic_BP = st.number_input("Diastolic BP", 30.0, 150.0, 80.0)
    Heart_Rate = st.number_input("Heart Rate", 30.0, 200.0, 75.0)
    Temperature = st.number_input("Temperature", 30.0, 45.0, 37.0)
    Respiratory_Rate = st.number_input("Respiratory Rate", 5.0, 60.0, 18.0)

with col2:
    st.header("👤 Demographics & Insurance")
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Insurance_Company = st.text_input("Insurance Company", "Daman")
    Insurance_Plan = st.text_input("Insurance Plan", "Basic")
    ICD_Code = st.text_input("ICD Code", "S72.001A")

with col3:
    st.header("📝 CPT, Notes & Prediction")
    CPT_Code = st.text_input("CPT Code", "99283")
    Clinical_Notes = st.text_area("Clinical Notes", "Patient reported chest pain and shortness of breath.")

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
            "Insurance_Plan": Insurance_Plan,
            "Clinical_Notes": Clinical_Notes
        }
        result = predict_claim(input_data, Clinical_Notes)
        st.success(f"📌 Predicted Claim Status: **{result}**")

