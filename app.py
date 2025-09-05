# ==============================
# 📦 Imports
# ==============================
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

try:
    import streamlit as st
except ImportError:
    st = None

# ==============================
# 🔧 CONFIG
# ==============================
RUN_MODE = "streamlit"   # "kaggle" for testing, "streamlit" for deployment
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

def match_input_to_encoder(encoder, value: str):
    """Case-insensitive + first-word matching for company/plan."""
    value_clean = value.strip().split()[0].lower()
    classes_clean = [c.lower().split()[0] for c in encoder.classes_]

    if value_clean in classes_clean:
        matched = encoder.classes_[classes_clean.index(value_clean)]
        return encoder.transform([matched])[0]
    else:
        # fallback: map to first known class
        return encoder.transform([encoder.classes_[0]])[0]

# ==============================
# LOAD + PREPROCESS
# ==============================
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    print(f"✅ Data loaded: {df.shape}")

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

# ==============================
# TRAINING
# ==============================
def train_and_save_model():
    X, y, encoders, vectorizer = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training Random Forest model...")
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

    print(f"✅ Model trained & saved (Accuracy: {acc:.3f})")
    print(report)
    print("Confusion Matrix:\n", cm)

    return model, encoders, vectorizer

# ==============================
# PREDICTION
# ==============================
def predict_claim(input_data):
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        return f"❌ Error loading model or encoders: {e}"

    df_input = pd.DataFrame([input_data])

    # Separate categorical and text
    categorical_features = ["Insurance_Company", "Insurance_Plan"]
    text_features = ["ICD_Code", "Clinical_Notes"]

    try:
        # Encode categorical (case-insensitive + partial match)
        for col in categorical_features:
            df_input[col] = df_input[col].apply(
                lambda x: match_input_to_encoder(encoders[col], str(x))
            )
    except Exception:
        # If encoding totally fails → fallback message
        return "⚠️ New company/plan detected. Please retrain model."

    # Vectorize ICD + notes
    text_data = df_input[text_features].astype(str).agg(" ".join, axis=1)
    text_vectorized = vectorizer.transform(text_data)

    # Merge with numeric
    df_input = df_input.drop(columns=text_features)
    X_numeric = df_input.values.astype(float)  # force numeric dtype
    X_input = hstack([X_numeric, text_vectorized])


    try:
        pred = model.predict(X_input)[0]
        return "✅ Approved" if pred == 0 else "❌ Denied"
    except Exception as e:
        return f"⚠️ Could not predict due to error: {e}"

# ==============================
# 🚀 MAIN EXECUTION (Kaggle)
# ==============================
if RUN_MODE == "kaggle":
    model, encoders, vectorizer = train_and_save_model()

    sample_input = {
        "Age": 45.5, "Systolic_BP": 120.2, "Diastolic_BP": 79.8,
        "Heart_Rate": 78.0, "Temperature": 37.5, "Respiratory_Rate": 18.2,
        "Gender": "Male", "CPT_Code": "99283", "Insurance_Company": "daman",
        "Insurance_Plan": "basic", 
        "ICD_Code": "I10 E11 Z79",   # multiple ICD codes joined
        "Clinical_Notes": "Patient reported chest pain and high blood pressure."
    }
    pred = predict_claim(sample_input)
    print(f"🔮 Prediction for sample input: {pred}")

# ==============================
# 🌐 STREAMLIT APP (only runs if deployed)
# ==============================
elif RUN_MODE == "streamlit":
    st.title("🏥 UAE Claim Approval Prediction (Random Forest + TF-IDF)")

    if st.button("Train/Reload Model"):
        train_and_save_model()

    st.header("Enter Claim Details")
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=40.0, step=0.1)
    sys_bp = st.number_input("Systolic BP", min_value=80.0, max_value=200.0, value=120.0, step=0.1)
    dia_bp = st.number_input("Diastolic BP", min_value=40.0, max_value=120.0, value=80.0, step=0.1)
    hr = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
    temp = st.number_input("Temperature", min_value=30.0, max_value=42.0, value=37.0, step=0.1)
    rr = st.number_input("Respiratory Rate", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
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
        icd_text = " ".join(icd_codes)

        input_data = {
            "Age": float(age), "Systolic_BP": float(sys_bp), "Diastolic_BP": float(dia_bp),
            "Heart_Rate": float(hr), "Temperature": float(temp), "Respiratory_Rate": float(rr),
            "Gender": gender, "CPT_Code": cpt,
            "Insurance_Company": company, "Insurance_Plan": plan,
            "ICD_Code": icd_text, "Clinical_Notes": notes
        }
        pred = predict_claim(input_data)
        st.subheader(f"Prediction: {pred}")

