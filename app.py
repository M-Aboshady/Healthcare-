import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec
from tensorflow.keras import backend as K

# --- FILE PATHS ---
file_path = "/kaggle/input/balanced-34/synthetic_claims_uae_capitalized_3.csv"
MODEL_PATH = "er_claim_model_w2v.h5"
ENCODERS_PATH = "er_encoders_w2v.joblib"
TOKENIZER_PATH = "er_notes_tokenizer_w2v.joblib"
WORD2VEC_MODEL_PATH = "er_word2vec_model.bin"

# --- MAPPING FOR TARGET VARIABLE ---
CLAIM_STATUS_MAP = {'Approved': 0, 'Rejected': 1}
CLASS_LABELS = ['Approved', 'Rejected']



def load_preprocess_and_train_model(file_path):
    """
    Loads, preprocesses, and trains a Wide and Deep neural network with Word2Vec embeddings.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the path is correct.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}. Please check your CSV file format.")
        return None, None, None, None, None

    print("📥 Data loaded:", df.shape)

    # Step 1 - Define features and target
    numerical_features = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    categorical_features = ['Gender', 'CPT_Code', 'ICD_Code', 'Insurance_Company', 'Insurance_Plan']
    text_feature = 'Clinical_Notes'
    target = 'Claim_Status'

    required_cols = numerical_features + categorical_features + [text_feature, target]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in CSV: {', '.join(missing)}")
        return None, None, None, None, None

    df_cleaned = df.dropna(subset=numerical_features + categorical_features + [text_feature, target])
    
    # Step 2 - Encode categorical features
    df_encoded = df_cleaned.copy()
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Step 3 - Encode target
    
    y = df_encoded[target].map(CLAIM_STATUS_MAP).astype(int)

    # Step 4 - Prepare text data for Word2Vec
    documents = [str(note).lower().split() for note in df_encoded[text_feature]]
    
    # Train Word2Vec model
    word2vec_model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.save(WORD2VEC_MODEL_PATH)
    
    # Create a tokenizer and prepare sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_encoded[text_feature].astype(str))
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    
    max_sequence_length = 50
    X_text_sequences = tokenizer.texts_to_sequences(df_encoded[text_feature].astype(str))
    X_text_padded = pad_sequences(X_text_sequences, maxlen=max_sequence_length, padding='post')
    
    # Create embedding matrix
    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    # Step 5 - Structured features matrix (convert to array)
    X_struct_full = df_encoded[numerical_features + categorical_features].values
    
    # Step 6 - Train/test split (stratified)
    train_idx, test_idx = train_test_split(
        np.arange(len(df_encoded)), test_size=0.2, random_state=42, stratify=y
    )

    Xs_train, Xs_test = X_struct_full[train_idx], X_struct_full[test_idx]
    Xt_train, Xt_test = X_text_padded[train_idx], X_text_padded[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Step 7 - Build and train the Wide and Deep Neural Network with Word2Vec
    
    # Wide Path (structured features)
    structured_input = Input(shape=(Xs_train.shape[1],), name='structured_input')
    wide_output = Dense(1, activation=None, name='wide_output')(structured_input)
    
    # Deep Path (text and structured features)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_sequence_length,
        trainable=False
    )(text_input)
    text_embedding_avg = GlobalAveragePooling1D()(embedding_layer)

    combined_deep = Concatenate()([structured_input, text_embedding_avg])
    deep_layer1 = Dense(516, activation='relu')(combined_deep)
    deep_layer1_dropout = Dropout(0.3)(deep_layer1)
    deep_layer2 = Dense(256, activation='relu')(deep_layer1_dropout)
    deep_layer2_dropout = Dropout(0.3)(deep_layer2)
    deep_layer3 = Dense(128, activation='relu')(deep_layer2_dropout)
    deep_layer3_dropout = Dropout(0.3)(deep_layer3)
    deep_layer4 = Dense(64, activation='relu')(deep_layer3_dropout)
    deep_layer4_dropout = Dropout(0.3)(deep_layer4)
    deep_layer5 = Dense(64, activation='relu')(deep_layer4_dropout)
    deep_layer5_dropout = Dropout(0.3)(deep_layer5)
    deep_layer6 = Dense(64, activation='relu')(deep_layer5_dropout)
    deep_output = Dense(1, activation=None, name='deep_output')(deep_layer6)
    
    # Final output
    final_output = Concatenate()([wide_output, deep_output])
    final_prediction = Dense(1, activation='sigmoid')(final_output)

    model = Model(inputs=[structured_input, text_input], outputs=final_prediction)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    history = model.fit(
        {'structured_input': Xs_train, 'text_input': Xt_train},
        y_train,
        epochs=70,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    print("🤖 Wide and Deep model with Word2Vec trained successfully")

    # Step 8 - Evaluate and save artifacts
    y_pred_prob = model.predict([Xs_test, Xt_test], verbose=0).flatten()
    y_pred = (y_pred_prob > 0.61).astype(int)###0.537 

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    prec_rejected = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_rejected = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_rejected = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print("✅ Validation Accuracy:", acc)
    print("Precision (Rejected):", prec_rejected)
    print("Recall (Rejected):", recall_rejected)
    print("F1-score (Rejected):", f1_rejected)
    print(report)

    model.save(MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    
    return model, encoders, tokenizer, (Xs_test, Xt_test, y_test, y_pred, acc, report, cm)

# --- SAFE ENCODER FUNCTION ---
def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        if "Unknown" not in le.classes_:
            le_classes = list(le.classes_)
            le_classes.append("Unknown")
            le.classes_ = np.array(le_classes)
        return le.transform(["Unknown"])[0]

# --- PREDICTION FUNCTION ---
def predict_claim(input_data, clinical_note):
    try:
        model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})
        encoders = joblib.load(ENCODERS_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
    except FileNotFoundError as e:
        print(f"Prediction artifacts not found. Please ensure the model has been trained. Error: {e}")
        return "Error"
    except Exception as e:
        print(f"An error occurred while loading model artifacts. Error: {e}")
        return "Error"

    numerical_features = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 'Respiratory_Rate']
    categorical_features = ['Gender', 'CPT_Code', 'ICD_Code', 'Insurance_Company', 'Insurance_Plan']
    
    df_input = pd.DataFrame([input_data])
    
    for col in categorical_features:
        if col in encoders:
            le = encoders[col]
            df_input[col] = df_input[col].apply(lambda x: safe_transform(le, str(x)))
    
    X_struct_input = df_input[numerical_features + categorical_features].values
    
    # Process text input
    max_sequence_length = 50
    X_text_input_sequence = tokenizer.texts_to_sequences([clinical_note])
    X_text_input_padded = pad_sequences(X_text_input_sequence, maxlen=max_sequence_length, padding='post')
    
    prediction_prob = model.predict([X_struct_input, X_text_input_padded], verbose=0)[0][0]
    prediction = int(prediction_prob > 0.61)##0.537

    return CLASS_LABELS[prediction]

if __name__ == "__main__":
    try:
        model, encoders, tokenizer, results = load_preprocess_and_train_model(file_path)
        Xs_test, Xt_test, y_test, y_pred, acc, report, cm = results
        print("🏁 Done. Model trained and evaluated.")

        print("\n📊 Model Evaluation Results")
        print(f"**Accuracy:** {acc:.2f}")
        print("Classification Report:")
        print(report)
        
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="ER Claim Prediction App", layout="wide")
st.title("🏥 Emergency Room Claim Status Prediction")
st.markdown("Enter the patient's vitals and clinical notes to predict if their claim will be **Approved** or **Rejected**.")

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
        icd_code = st.text_input("ICD Code", "I10")
        insurance_company = st.text_input("Insurance Company", "Daman")
    with col4:
        insurance_plan = st.text_input("Insurance Plan", "Basic")
    
    st.header("Clinical Notes")
    clinical_note = st.text_area("Clinical Notes", "Patient stable, normal findings")
    
    submitted = st.form_submit_button("Predict Claim Status")

if submitted:
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



