# tfidf_chatbot.py
import json
import random
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load and preprocess dataset
# -----------------------------
def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Load intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions, responses, categories = [], [], []

for intent in data.get("intents", []):
    tag = intent.get("tag", "unknown")
    for pattern in intent.get("patterns", []):
        questions.append(clean_text(pattern))
        responses.append(intent.get("responses", ["Sorry, I don't understand."]))
        categories.append(tag)

# -----------------------------
# Train TF-IDF vectorizer
# -----------------------------
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(questions)

# -----------------------------
# Chatbot response function
# -----------------------------
def tfidf_chatbot(user_input: str, threshold: float = 0.2):
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    sims = cosine_similarity(user_vec, X_tfidf)[0]

    best_idx = sims.argmax()
    best_score = sims[best_idx]
    
    if best_score < threshold:
        return random.choice([
           "Hmm, I'm not sure I follow. Could you explain in another way?",
            "I didn't quite understand that. Can you rephrase?",
            "Sorry, I'm not sure. Maybe you can try asking differently?",
        ]), "unknown"
    
    return random.choice(responses[best_idx]), categories[best_idx]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="TF-IDF Chatbot", layout="centered")
st.title("🤖 TF-IDF Chatbot")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = [("Bot", "Hello! I'm your assistant. How can I help you today?")]

#input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here:", "")
    submitted = st.form_submit_button("Send")
    
if submitted and user_input.strip():
    st.session_state.history.append(("User", user_input))
    bot_reply, _ = tfidf_chatbot(user_input)
    st.session_state.history.append(("Bot", bot_reply))

# Display chat
chat_container = st.container()
with chat_container:
    for speaker, msg in st.session_state.history:
        if speaker == "User":
            st.markdown(
                f"<div style='text-align:right; background-color:#DCF8C6; padding:8px; border-radius:10px; margin:5px;'><b>{speaker}:</b> {msg}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left; background-color:#ECECEC; padding:8px; border-radius:10px; margin:5px;'><b>{speaker}:</b> {msg}</div>",
                unsafe_allow_html=True,
            )

# Auto-scroll
st.markdown("<div id='scroll-bottom'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <script>
        var scrollDiv = document.getElementById('scroll-bottom');
        scrollDiv.scrollIntoView({behavior: 'smooth'});
    </script>
    """,
    unsafe_allow_html=True
)