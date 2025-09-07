# tfidf_evaluation_app.py
import streamlit as st
import json, random, re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Load intents and train TF-IDF
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions, responses, categories = [], [], []

for intent in data.get("intents", []):
    tag = intent.get("tag", "unknown")
    for pattern in intent.get("patterns", []):
        questions.append(clean_text(pattern))
        responses.append(intent.get("responses", ["Sorry, I don't understand."]))
        categories.append(tag)

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(questions)

def tfidf_chatbot(user_input: str, threshold: float = 0.2):
    user_vec = vectorizer.transform([clean_text(user_input)])
    sims = cosine_similarity(user_vec, X_tfidf)[0]
    best_idx = sims.argmax()
    if sims[best_idx] < threshold:
        return random.choice([
            "Hmm, I'm not sure I follow. Can you rephrase?",
            "Sorry, I didn't quite understand that."
        ]), "unknown"
    return random.choice(responses[best_idx]), categories[best_idx]

# -----------------------------
# Evaluation
# -----------------------------
st.set_page_config(page_title="TF-IDF Chatbot Evaluation", layout="centered")
st.title("📊 TF-IDF Chatbot Evaluation")

train_pairs, test_pairs = train_test_split(list(zip(questions, responses, categories)), test_size=0.3, random_state=42)

y_true, y_pred = [], []
y_true_cat, y_pred_cat = [], []

for q, r, c in test_pairs:
    pred_r, pred_c = tfidf_chatbot(q)
    y_true.append(r[0])
    y_pred.append(pred_r)
    y_true_cat.append(c)
    y_pred_cat.append(pred_c)

# Response-level metrics
resp_accuracy = accuracy_score(y_true, y_pred) * 100
resp_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
resp_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
resp_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

resp_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1 Score"],
    "Score (%)": [resp_accuracy, resp_precision, resp_recall, resp_f1]
})
st.subheader("Response-Level Metrics")
st.dataframe(resp_df)

# Category-level metrics
cat_accuracy = accuracy_score(y_true_cat, y_pred_cat) * 100
cat_precision = precision_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0) * 100
cat_recall = recall_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0) * 100
cat_f1 = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0) * 100

cat_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1 Score"],
    "Score (%)": [cat_accuracy, cat_precision, cat_recall, cat_f1]
})
st.subheader("Category-Level Metrics")
st.dataframe(cat_df)

# Plot charts
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].bar(resp_df["Metric"], resp_df["Score (%)"], color="skyblue")
ax[0].set_ylim(0,100)
ax[0].set_title("Response-Level Metrics")

ax[1].bar(cat_df["Metric"], cat_df["Score (%)"], color="lightgreen")
ax[1].set_ylim(0,100)
ax[1].set_title("Category-Level Metrics")

st.pyplot(fig)
