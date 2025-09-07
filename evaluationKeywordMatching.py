import streamlit as st
import json
import csv
import re
import random
import unicodedata
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load keyword matching logic
# -------------------------
# Copy this from your keyword_matching.py (without Tkinter)
qa_pairs = []

def clean_text_dataset(text):
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'"}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii").strip()

# Load intents.json
try:
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for intent in data.get("intents", []):
            category = intent.get("tag", "unknown")
            for pattern in intent.get("patterns", []):
                responses = [clean_text_dataset(r) for r in intent.get("responses", [])]
                qa_pairs.append((pattern.lower(), responses, category))
except:
    pass

# Load FAQ CSV
try:
    with open("Mental_Health_FAQ.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("questions","").lower()
            a = clean_text_dataset(row.get("answers",""))
            if q and a: qa_pairs.append((q,[a],"FAQ"))
except:
    pass

if not qa_pairs:
    qa_pairs = [
        ("hello", ["Hello! How are you feeling today?"], "greeting"),
        ("i feel sad", ["I'm sorry to hear that. Do you want to talk more?"], "depression")
    ]

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

def keyword_overlap(user_input, question):
    user_words = set(clean_text(user_input).split())
    question_words = set(clean_text(question).split())
    return len(user_words & question_words) / max(1, len(user_words))

def best_match(user_input, qa_set=qa_pairs):
    best_score = 0
    best_responses = None
    best_category = "unknown"
    
    for q, r, c in qa_set:
        score = keyword_overlap(user_input, q)
        if score > best_score:
            best_score = score
            best_responses = r
            best_category = c
    
    if best_score < 0.1:
        return random.choice([
            "Hmm, I'm not sure I follow. Could you explain in another way?",
            "I didn't quite understand that. Can you rephrase?",
            "Sorry, I'm not sure. Maybe you can try asking differently?",
        ]), "unknown"
    
    return random.choice(best_responses), best_category

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Keyword Matching Evaluation", layout="wide")
st.title("📊 Keyword Matching Chatbot Evaluation")

# Train/test split
train_pairs, test_pairs = train_test_split(qa_pairs, test_size=0.3, random_state=42)

y_true, y_pred = [], []
y_true_cat, y_pred_cat = [], []

for question, correct_answers, true_category in test_pairs:
    predicted, pred_category = best_match(question, qa_set=train_pairs)
    y_true.append(correct_answers[0])
    y_pred.append(predicted)
    y_true_cat.append(true_category)
    y_pred_cat.append(pred_category)

# Response-level evaluation
encoder = LabelEncoder()
all_responses = list(set(y_true + y_pred))
encoder.fit(all_responses)

y_true_encoded = encoder.transform(y_true)
y_pred_encoded = encoder.transform(y_pred)

resp_accuracy = accuracy_score(y_true_encoded, y_pred_encoded) * 100
resp_precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0) * 100
resp_recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0) * 100
resp_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0) * 100

resp_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score (%)": [resp_accuracy, resp_precision, resp_recall, resp_f1]
})

# Category-level evaluation
cat_encoder = LabelEncoder()
all_categories = list(set(y_true_cat + y_pred_cat))
cat_encoder.fit(all_categories)

y_true_cat_encoded = cat_encoder.transform(y_true_cat)
y_pred_cat_encoded = cat_encoder.transform(y_pred_cat)

cat_accuracy = accuracy_score(y_true_cat_encoded, y_pred_cat_encoded) * 100
cat_precision = precision_score(y_true_cat_encoded, y_pred_cat_encoded, average='weighted', zero_division=0) * 100
cat_recall = recall_score(y_true_cat_encoded, y_pred_cat_encoded, average='weighted', zero_division=0) * 100
cat_f1 = f1_score(y_true_cat_encoded, y_pred_cat_encoded, average='weighted', zero_division=0) * 100

cat_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score (%)": [cat_accuracy, cat_precision, cat_recall, cat_f1]
})

# -------------------------
# Display results in Streamlit
# -------------------------
st.subheader("Response-Level Evaluation")
st.dataframe(resp_results)
st.bar_chart(resp_results.set_index("Metric"))

st.subheader("Category-Level Evaluation")
st.dataframe(cat_results)
st.bar_chart(cat_results.set_index("Metric"))
