# --- evaluationBert.py ---
import streamlit as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="BERT Chatbot Evaluation 🤖", layout="centered")
st.title("🤖 BERT Chatbot Evaluation")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# Training patterns & labels
# -------------------------------
patterns = [
    "Hello",
    "I'm feeling sad",
    "I'm stressed out",
    "I'm depressed",
    "I want to end my life",
    "What can you do?",
    "Who made you?",
    "Can you do impossible things?"
]
labels = ["greeting", "sad", "stressed", "depressed", "suicide", "skill", "creation", "none"]

test_queries = [
    ("Hi", "greeting"),
    ("I feel sad", "sad"),
    ("I feel stuck", "stressed"),
    ("I can't take it anymore", "depressed"),
    ("I want to die", "suicide"),
    ("What can you do", "skill"),
    ("Who created you", "creation"),
    ("Can you fly?", "none")
]

# -------------------------------
# Encode patterns
# -------------------------------
pattern_embeddings = model.encode(patterns, convert_to_tensor=True)

# -------------------------------
# Generate predictions
# -------------------------------
y_true, y_pred = [], []
for query, true_tag in test_queries:
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_emb, pattern_embeddings)[0]
    best_idx = int(torch.argmax(sims))
    pred_tag = labels[best_idx]

    y_true.append(true_tag)
    y_pred.append(pred_tag)

# -------------------------------
# Overall evaluation
# -------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

overall_metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}

st.subheader("📊 Overall Metrics")
st.dataframe(
    {k: [f"{v*100:.2f}%"] for k, v in overall_metrics.items()},
    use_container_width=True
)

# --- Plot Overall Metrics ---
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(overall_metrics.keys(), overall_metrics.values(), color="skyblue")
ax.set_ylim(0,1)
ax.set_title("Response-Level Evaluation (Overall Metrics)")
for i, v in enumerate(overall_metrics.values()):
    ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")
st.pyplot(fig)

# -------------------------------
# Category-Level Evaluation
# -------------------------------
st.subheader("📊 Category-Level Metrics")

categories = sorted(set(y_true))  # unique intents in test set
precision_cat, recall_cat, f1_cat, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=categories, zero_division=0
)

# Accuracy per category
accuracy_cat = []
for category in categories:
    y_true_binary = [1 if y == category else 0 for y in y_true]
    y_pred_binary = [1 if y == category else 0 for y in y_pred]
    cat_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    accuracy_cat.append(cat_accuracy)

# --- Grouped bar chart ---
x = np.arange(len(categories))
width = 0.2
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - 1.5*width, accuracy_cat, width, label="Accuracy")
ax.bar(x - 0.5*width, precision_cat, width, label="Precision")
ax.bar(x + 0.5*width, recall_cat, width, label="Recall")
ax.bar(x + 1.5*width, f1_cat, width, label="F1-score")

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45)
ax.set_ylim(0,1)
ax.set_title("Category-Level Evaluation (Per Intent)")
ax.legend()

# Add % labels
for i in range(len(categories)):
    ax.text(i - 1.5*width, accuracy_cat[i] + 0.02, f"{accuracy_cat[i]*100:.1f}%", ha="center", fontsize=8)
    ax.text(i - 0.5*width, precision_cat[i] + 0.02, f"{precision_cat[i]*100:.1f}%", ha="center", fontsize=8)
    ax.text(i + 0.5*width, recall_cat[i] + 0.02, f"{recall_cat[i]*100:.1f}%", ha="center", fontsize=8)
    ax.text(i + 1.5*width, f1_cat[i] + 0.02, f"{f1_cat[i]*100:.1f}%", ha="center", fontsize=8)

st.pyplot(fig)

# -------------------------------
# Detailed report
# -------------------------------
st.subheader("📑 Detailed Report")

# --- Classification Report as Table ---
st.markdown("**Classification Report**")
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

# --- Confusion Matrix as Table ---
st.markdown("**Confusion Matrix**")
cm = confusion_matrix(y_true, y_pred, labels=categories)
cm_df = pd.DataFrame(cm, index=categories, columns=categories)
st.dataframe(cm_df, use_container_width=True)

# --- Per-category metrics summary ---
st.markdown("**Per-Category Metrics**")
per_cat_df = pd.DataFrame({
    "Category": categories,
    "Accuracy": [f"{a*100:.2f}%" for a in accuracy_cat],
    "Precision": [f"{p*100:.2f}%" for p in precision_cat],
    "Recall": [f"{r*100:.2f}%" for r in recall_cat],
    "F1-score": [f"{f*100:.2f}%" for f in f1_cat]
})
st.dataframe(per_cat_df, use_container_width=True)

