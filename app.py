import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import re

from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from gensim.models import Word2Vec

# Deep Learning specific imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# --- FILE PATHS & CONFIG ---
DATA_PATH = "/kaggle/input/multi-1/synthetic_claims_uae_multi_icd.csv"
MODEL_PATH = "dl_claim_model.h5"
ARTIFACTS_PATH = "dl_model_artifacts.joblib"
W2V_ICD_PATH = "w2v_icd.model"
W2V_NOTES_PATH = "w2v_notes.model"

CLAIM_STATUS_MAP = {'Approved': 1, 'Rejected': 0}
CLASS_LABELS = ['Rejected', 'Approved']

# --- UTILITIES ---
def _tokenize(text: str, delimiter=',') -> List[str]:
    text = str(text).lower()
    return [t.strip() for t in re.split(delimiter, text) if t.strip()]

def _mean_embed(items: List[str], w2v: Word2Vec, dim: int) -> np.ndarray:
    vecs = [w2v.wv[it] for it in items if it in w2v.wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

def _safe_label_transform(le: LabelEncoder, value: str) -> int:
    value = str(value)
    if value in le.classes_:
        return int(le.transform([value])[0])
    classes = list(le.classes_)
    if "Unknown" not in classes:
        classes.append("Unknown")
        le.classes_ = np.array(classes)
    return int(le.transform(["Unknown"])[0])

# --- DATA PREPROCESSING & MODEL TRAINING ---
@st.cache_resource
def train_model():
    """Trains a new deep learning model and saves all artifacts."""
    st.info("Model artifacts not found. Initiating model training...")
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        df = df.dropna(subset=['Age', 'Claim_Status'])  # Minimal cleaning
    except FileNotFoundError:
        st.error(f"Error: The data file '{DATA_PATH}' was not found.")
        st.stop()

    st.write("📥 Data loaded:", df.shape)

    # Convert target to integers
    df['Claim_Status_int'] = df['Claim_Status'].apply(
        lambda x: CLAIM_STATUS_MAP.get(str(x).capitalize(), -1)
    )
    df = df[df['Claim_Status_int'] != -1]
    y = df['Claim_Status_int'].values

    # Define features
    num_cols = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    cat_cols = ['Gender', 'CPT_Code', 'Insurance_Company', 'Insurance_Plan']
    icd_col = 'ICD_Code'
    notes_col = 'Clinical_Notes'

    # --- Preprocess Categorical Features ---
    encoders = {c: LabelEncoder().fit(df[c].astype(str)) for c in cat_cols}
    X_cat_encoded = {c: encoders[c].transform(df[c].astype(str)) for c in cat_cols}

    # --- Preprocess Text Features with Word2Vec ---
    icd_lists = df[icd_col].apply(lambda x: _tokenize(x, ',')).tolist()
    w2v_icd = Word2Vec(icd_lists, vector_size=50, window=5, min_count=1, workers=4)
    w2v_icd.save(W2V_ICD_PATH)
    
    notes_tokens = df[notes_col].apply(lambda x: _tokenize(x, ' ')).tolist()
    w2v_notes = Word2Vec(notes_tokens, vector_size=50, window=5, min_count=1, workers=4)
    w2v_notes.save(W2V_NOTES_PATH)

    X_icd = np.vstack([_mean_embed(lst, w2v_icd, 50) for lst in icd_lists])
    X_notes = np.vstack([_mean_embed(toks, w2v_notes, 50) for toks in notes_tokens])

    # --- Prepare Data for Deep Learning ---
    X_num = df[num_cols].values
    X_cat = np.hstack([X_cat_encoded[c].reshape(-1, 1) for c in cat_cols])
    
    # Concatenate all features
    X_combined = np.hstack([X_num, X_cat, X_icd, X_notes])
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Build Functional Model ---
    num_input = Input(shape=(X_num.shape[1],), name='num_input')
    cat_input = Input(shape=(X_cat.shape[1],), name='cat_input')
    icd_input = Input(shape=(X_icd.shape[1],), name='icd_input')
    notes_input = Input(shape=(X_notes.shape[1],), name='notes_input')

    merged = Concatenate()([num_input, cat_input, icd_input, notes_input])
    
    dense1 = Dense(256, activation='relu')(merged)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    output = Dense(1, activation='sigmoid')(dropout2)
    
    model = Model(inputs=[num_input, cat_input, icd_input, notes_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Train the model
    model.fit(
        {'num_input': X_train[:, :len(num_cols)],
         'cat_input': X_train[:, len(num_cols):len(num_cols)+len(cat_cols)],
         'icd_input': X_train[:, len(num_cols)+len(cat_cols):len(num_cols)+len(cat_cols)+50],
         'notes_input': X_train[:, len(num_cols)+len(cat_cols)+50:]},
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate the model
    y_pred_prob = model.predict([X_test[:, :len(num_cols)], X_test[:, len(num_cols):len(num_cols)+len(cat_cols)],
                                 X_test[:, len(num_cols)+len(cat_cols):len(num_cols)+len(cat_cols)+50],
                                 X_test[:, len(num_cols)+len(cat_cols)+50:]], verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0)
    
    # Save artifacts
    model.save(MODEL_PATH)
    artifacts = {
        'encoders': encoders,
        'icd_w2v': w2v_icd,
        'notes_w2v': w2v_notes,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'icd_col': icd_col,
        'notes_col': notes_col,
        'known_companies': set(df['Insurance_Company'].unique()),
        'known_plans': set(df['Insurance_Plan'].unique()),
        'val_accuracy': acc,
        'val_report': report
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    st.success("🤖 Deep learning model trained and saved successfully!")
    return artifacts

# --- PREDICTION FUNCTION ---
def predict_claim(payload: Dict) -> Tuple[str, float]:
    """Loads artifacts and makes a single prediction."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        artifacts = joblib.load(ARTIFACTS_PATH)
        w2v_icd = Word2Vec.load(W2V_ICD_PATH)
        w2v_notes = Word2Vec.load(W2V_NOTES_PATH)
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return "Error", 0.0

    num_cols = artifacts['num_cols']
    cat_cols = artifacts['cat_cols']
    encoders = artifacts['encoders']
    icd_col = artifacts['icd_col']
    notes_col = artifacts['notes_col']

    # Preprocess inputs
    X_num_input = np.array([[float(payload.get(c, 0.0)) for c in num_cols]], dtype=np.float32)
    X_cat_input = np.array([[_safe_label_transform(encoders[c], payload.get(c, 'Unknown')) for c in cat_cols]], dtype=np.float32)
    
    icd_tokens = _tokenize(payload.get(icd_col, ''), ',')
    X_icd_input = _mean_embed(icd_tokens, w2v_icd, 50).reshape(1, -1)
    
    notes_tokens = _tokenize(payload.get(notes_col, ''))
    X_notes_input = _mean_embed(notes_tokens, w2v_notes, 50).reshape(1, -1)

    # Predict
    prob = model.predict(
        {'num_input': X_num_input, 'cat_input': X_cat_input, 'icd_input': X_icd_input, 'notes_input': X_notes_input}
    )[0, 0]
    
    pred_label = CLASS_LABELS[int(prob > 0.5)]
    return pred_label, prob

# --- MAIN APP EXECUTION ---
if __name__ == "__main__":
    st.title("🏥 UAE ER Claim Approval Prediction App (Functional DL)")

    # Check for model and train if needed
    if not os.path.exists(MODEL_PATH):
        artifacts = train_model()
    else:
        try:
            artifacts = joblib.load(ARTIFACTS_PATH)
            st.success(f"Model loaded. Validation accuracy: **{artifacts['val_accuracy']:.3f}**")
        except Exception:
            st.warning("Failed to load artifacts. Retraining the model.")
            artifacts = train_model()
            
    if artifacts is None:
        st.stop()

    # Input fields for the user
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.header("🫀 Vital Signs")
        age = st.number_input("Age", 0, 120, 30)
        heart_rate = st.number_input("Heart Rate", 30, 200, 75)
        respiratory_rate = st.number_input("Respiratory Rate", 5, 60, 18)
    
    with col2:
        st.header("👤 Demographics & Insurance")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        insurance_company = st.text_input("Insurance Company", "Daman")
        insurance_plan = st.text_input("Insurance Plan", "Basic")
        cpt_code = st.text_input("CPT Code", "99283")

    with col3:
        st.header("📝 Claim Details")
        systolic_bp = st.number_input("Systolic BP", 50, 250, 120)
        diastolic_bp = st.number_input("Diastolic BP", 30, 150, 80)
        temperature = st.number_input("Temperature", 30, 45, 37)
        icd_code = st.text_input("ICD Code (e.g., 'S72.001A,I10')", "S72.001A")
        clinical_notes = st.text_area("Clinical Notes", "Patient reported chest pain and shortness of breath.", height=100)

    if st.button("🔮 Predict Claim Status"):
        if insurance_company not in artifacts['known_companies'] or insurance_plan not in artifacts['known_plans']:
            st.warning("⚠️ New insurance or new plan detected. Prediction stopped.")
            st.info("The model cannot reliably predict for new insurance data as it was not in the training set.")
        else:
            payload = {
                'Age': age, 'Gender': gender, 'Heart_Rate': heart_rate, 'Respiratory_Rate': respiratory_rate,
                'Systolic_BP': systolic_bp, 'Diastolic_BP': diastolic_bp, 'Temperature': temperature,
                'CPT_Code': cpt_code, 'ICD_Code': icd_code,
                'Insurance_Company': insurance_company, 'Insurance_Plan': insurance_plan,
                'Clinical_Notes': clinical_notes
            }
            pred_label, prob = predict_claim(payload)
            st.success(f"📌 Predicted Claim Status: **{pred_label}** (Probability: {prob:.3f})")

    # Display validation report
    if 'val_report' in artifacts:
        with st.expander("📊 Validation Report"):
            st.text(artifacts['val_report'])

       
