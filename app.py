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
RUN_MODE = "kaggle"   # change to "streamlit" only when deploying
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

