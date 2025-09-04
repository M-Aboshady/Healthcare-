import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
import os

# Only import Streamlit if in streamlit mode
try:
    import streamlit as st
except ImportError:
    st = None

# ==============================
# 🔧 CONFIG
# ==============================
RUN_MODE = "kaggle"   # change to "streamlit" when deploying
DATA_PATH = "synthetic_claims_uae_multi_icd.csv"
MODEL_PATH = "er_claim_model_rf.joblib"
ENCODERS_PATH = "er_encoders_rf.joblib"
VECTORIZER_PATH = "er_notes_vectorizer_rf.joblib"

CLAIM_STATUS_MAP = {"Approved": 0, "Rejected": 1}
CLASS_LABELS = ["Approved", "Rejected"]


# ==============================
# UTILS
# ==============================
def safe_transform(le, value):
    """Safely transform a label if it's known, else map to first class."""
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return le.transform([le.classes_[0]])[0]


# ==============================
# LOAD + PREPROCESS
# ==============================
def load_and_preprocess_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = [col.strip().replace(" ", "_") for col in df.columns]
        if RUN_MODE == "kaggle":
            print(f"✅ Data loaded: {df.shape}")
        else:
            st.success(f"Data loaded: {df.shape}")

        numerical_features = [
            "Age", "Systolic_BP", "Diastolic_BP",
            "Heart_Rate", "Temperature", "Respiratory_Rate"
        ]
        categorical_features = [
            "Gender", "CPT_Code", "Insurance_Company", "Insurance_Plan"
        ]
        target = "Claim_Status"

        # Text features
        df["combined_text"] = df["ICD_Code"].fillna("") + " " + df["Clinical_Notes"].fillna("")
        df_cleaned = df.dropna(subset=numerical_features + categorical_features + ["combined_text", target])

        # Encode categorical
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
            encoders[col] = le

        # Encode text
        vectorizer = TfidfVectorizer(max_features=500)
        X_text = vectorizer.fit_transform(df_cleaned["combined_text"])

        # Target
        y_encoded = df_cleaned[target].map(CLAIM_STATUS_MAP).astype(int)

        # Structured features
        X_structured = df_cleaned[numerical_features + categorical_features]

        # Combined features
        X_combined = hstack([X_structured.values, X_text])

        return X_combined, y_encoded, encoders, vectorizer
    except Exception as e:
        if RUN_MODE == "kaggle":
            print(f"❌ Error loading data: {e}")
        else:
            st.error(f"Error loading data: {e}")
        return None, None, None, None


# ==============================
# TRAINING
# ==============================
def train_and_save_model():
    X, y, encoders, vectorizer = load_and_preprocess_data()
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if RUN_MODE == "kaggle":
        print(f"🚀 Training Random Forest model...")
    else:
        st.info("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    if RUN_MODE == "kaggle":
        print(f"✅ Model trained & saved (Accuracy: {acc:.3f})")
        print(report)
        print("Confusion Matrix:\n", cm)
    else:
        st.success(f"Model trained (Accuracy: {acc:.3f})")
        st.text(report)
        st.write("Confusion Matrix", cm)

    return model, encoders, vectorizer


# ==============================
# PREDICTION
# ==============================
def predict_claim(input_data):
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        return "Error", None

    numerical_features = ["Age", "Systolic_BP", "Diastolic_BP", "Heart_Rate", "Temperature", "Respiratory_Rate"]
    categorical_features = ["Gender", "CPT_Code", "Insurance_Company", "Insurance_Plan"]

    df_input = pd.DataFrame([input_data])

    # 🚨 New policy/plan check
    for col in ["Insurance_Company", "Insurance_Plan"]:
        le = encoders[col]
        if str(df_input[col].iloc[0]) not in le.classes_:
            msg = f"❌ New {col.replace('_', ' ')} '{df_input[col].iloc[0]}' not in training data. Retrain required."
            if RUN_MODE == "kaggle":
                print(msg)
            else:
                st.error(msg)
            return "Unknown", None

    # Encode categorical
    for col in categorical_features:
        df_input[col] = df_input[col].apply(lambda x: safe_transform(encoders[col], str(x)))

    # Text
    combined_text_input = str(input_data["ICD_Code"]) + " " + str(input_data["Clinical_Notes"])
    X_text = vectorizer.transform([combined_text_input])
    X_structured = df_input[numerical_features + categorical_features]
    X_input = hstack([X_structured.values, X_text])

    prediction = model.predict(X_input)[0]
    return CLASS_LABELS[prediction], model


# ==============================
# 🚀 MAIN EXECUTION
# ==============================
if RUN_MODE == "kaggle":
    # Train and test in Kaggle
    model, encoders, vectorizer = train_and_save_model()

    sample_input = {
        "Age": 45, "Systolic_BP": 120, "Diastolic_BP": 80,
        "Heart_Rate": 78, "Temperature": 37, "Respiratory_Rate": 18,
        "Gender": "Male", "CPT_Code": "99283", "Insurance_Company": "Daman",
        "Insurance_Plan": "Basic", "ICD_Code": "I10",
        "Clinical_Notes": "Patient reported chest pain and high blood pressure."
    }
    pred, _ = predict_claim(sample_input)
    print(f"🔮 Prediction for sample input: {pred}")

elif RUN_MODE == "streamlit":
    st.title("🏥 UAE Claim Approval Prediction (Random Forest + TF-IDF)")

    # Train/load model
    if st.button("Train/Reload Model"):
        train_and_save_model()

    # User inputs
    st.header("Enter Claim Details")
    age = st.number_input("Age", 0, 120, 40)
    sys_bp = st.number_input("Systolic BP", 80, 200, 120)
    dia_bp = st.number_input("Diastolic BP", 40, 120, 80)
    hr = st.number_input("Heart Rate", 30, 200, 75)
    temp = st.number_input("Temperature", 30, 42, 37)
    rr = st.number_input("Respiratory Rate", 5, 50, 18)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cpt = st.text_input("CPT Code", "99283")
    company = st.text_input("Insurance Company", "Daman")
    plan = st.text_input("Insurance Plan", "Basic")

    # Multi ICD entries (up to 4)
    st.subheader("ICD Codes (up to 4)")
    icd_codes = []
    for i in range(1, 5):
        code = st.text_input(f"ICD Code {i}", "" if i > 1 else "I10")
        if code.strip():
            icd_codes.append(code.strip())

    notes = st.text_area("Clinical Notes", "Patient has high BP and chest pain.")

    if st.button("Predict"):
        # Join ICD codes into a single string
        icd_text = " ".join(icd_codes)

        input_data = {
            "Age": age, "Systolic_BP": sys_bp, "Diastolic_BP": dia_bp,
            "Heart_Rate": hr, "Temperature": temp, "Respiratory_Rate": rr,
            "Gender": gender, "CPT_Code": cpt,
            "Insurance_Company": company, "Insurance_Plan": plan,
            "ICD_Code": icd_text, "Clinical_Notes": notes
        }
        pred, _ = predict_claim(input_data)
        st.subheader(f"Prediction: {pred}")
