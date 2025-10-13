# ==============================================
# Prescription Documentation Score Calculator
# Using ClinicalBERT (Manual Input Version)
# ==============================================

import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. Load Clinical-BERT Model
# =========================
@st.cache_resource
def load_clinical_bert():
    """
    Loads ClinicalBERT tokenizer and model.
    Cached to avoid reloading every time the app reruns.
    """
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tokenizer, model

tokenizer, model = load_clinical_bert()

# =========================
# 2. Helper Functions
# =========================

def sanitize_input(text):
    """Basic sanitization to prevent malicious input."""
    return text.replace("<", "&lt;").replace(">", "&gt;").strip()

def get_embedding(text, tokenizer, model):
    """Generate ClinicalBERT embedding for given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # CLS token

def calculate_documentation_score(icd_description, drug_name, tokenizer, model):
    """
    Calculates similarity between ICD and Drug text embeddings.
    Returns a score scaled between 0.1 and 1.0.
    """
    icd_description = sanitize_input(icd_description)
    drug_name = sanitize_input(drug_name)

    icd_emb = get_embedding(icd_description, tokenizer, model)
    drug_emb = get_embedding(drug_name, tokenizer, model)
    similarity = cosine_similarity(icd_emb, drug_emb)[0][0]

    # Normalize similarity to 0.1–1.0
    score = 0.1 + 0.9 * max(0, min(1, similarity))
    return round(score, 3)

# =========================
# 3. Streamlit App Layout
# =========================
st.set_page_config(page_title="Documentation Score Calculator", layout="wide")

st.title("🧾 Prescription Documentation Score Calculator")
st.markdown("""
This tool uses **ClinicalBERT** to evaluate how consistent a prescribed **Drug Name** is
with the given **ICD Code Description**, returning a Documentation Score between **0.1 and 1.0**.
""")

st.markdown("---")

# =========================
# 4. Manual Input Section
# =========================

st.subheader("Enter Prescription Information Manually")

col1, col2 = st.columns(2)

with col1:
    drug_name = st.text_area(
        "Generic Drug Name",
        height=150,
        placeholder="e.g., Metformin Hydrochloride 500mg",
        help="Enter the prescribed generic drug name.",
    )

with col2:
    icd_description = st.text_area(
        "ICD Code Description",
        height=150,
        placeholder="e.g., Type 2 diabetes mellitus without complications",
        help="Enter the ICD code description or diagnosis in English.",
    )

# Add some spacing
st.write("")
calculate = st.button("🔍 Calculate Documentation Score")

if calculate:
    if not drug_name.strip() or not icd_description.strip():
        st.error("Please fill in both fields before calculating.")
    else:
        with st.spinner("Analyzing with ClinicalBERT..."):
            score = calculate_documentation_score(icd_description, drug_name, tokenizer, model)

        st.success(f"✅ Documentation Score: **{score}**")
        st.progress(score)

        # Optional: Display interpretation
        if score >= 0.8:
            st.info("Excellent consistency — documentation appears complete and clinically aligned.")
        elif score >= 0.5:
            st.warning("Moderate consistency — consider reviewing the diagnosis-drug relation.")
        else:
            st.error("Low consistency — possible mismatch between ICD description and prescribed drug.")

# =========================
# 5. Security & Disclaimer
# =========================
st.markdown("---")
st.caption("""
🔒 **Security & Privacy Notes**
- All inputs are sanitized before processing.
- No uploaded data is stored or transmitted.
- ClinicalBERT is loaded locally from the Hugging Face repository.
- This demo is for educational and research use — not for clinical decision-making.
""")

