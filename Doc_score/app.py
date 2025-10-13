# ============================================
# 📘 Biomedical Semantic Similarity Scoring
# ============================================

from sentence_transformers import SentenceTransformer, util
import torch

# --------------------------------------------
# 1️⃣ Load the domain-specific biomedical model
# --------------------------------------------
# Recommended model for semantic similarity in clinical context
model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
model = SentenceTransformer(model_name)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# --------------------------------------------
# 2️⃣ Define similarity scoring function
# --------------------------------------------
def get_documentation_score(icd_description, drug_name):
    """
    Compute a similarity-based documentation consistency score
    between ICD description and Drug name (0.1 – 1.0).
    """

    # Add contextual hints to improve semantic distinction
    icd_text = f"ICD code description: {icd_description}"
    drug_text = f"Generic drug name: {drug_name}"

    # Encode both sentences
    embeddings = model.encode(
        [icd_text, drug_text],
        convert_to_tensor=True,
        device=device
    )

    # Compute cosine similarity
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Normalize the score to [0.1, 1.0]
    normalized_score = (cosine_score - 0.5) / (0.9 - 0.5)
    normalized_score = max(0.1, min(normalized_score, 1.0))

    return round(normalized_score, 3)

# --------------------------------------------
# 3️⃣ Test with examples
# --------------------------------------------
examples = [
    ("Type 2 Diabetes Mellitus", "Metformin"),
    ("Hypertension", "Amlodipine"),
    ("Fever", "Metformin"),
    ("Depression", "Sertraline"),
]

print("\n🩺 Prescription Documentation Consistency Test\n")
for icd, drug in examples:
    score = get_documentation_score(icd, drug)
    print(f"ICD: {icd:<30} | Drug: {drug:<15} → Score: {score}")
