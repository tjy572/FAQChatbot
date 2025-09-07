# bertChatbot.py
import json
import random
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Mental Health FAQ Chatbot 🤖", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

intents = {
    item["tag"]: {
        "patterns": item.get("patterns", []),
        "responses": item.get("responses", [])
    }
    for item in data.get("intents", [])
}

# -----------------------------
# Load Sentence Transformer Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2").to(device)

model = load_model()

# -----------------------------
# Prepare Embeddings
# -----------------------------
pattern_texts, labels = [], []
for tag, obj in intents.items():
    for pattern in obj["patterns"]:
        pattern_texts.append(pattern)
        labels.append(tag)

pattern_embeddings = model.encode(pattern_texts, convert_to_tensor=True)

# -----------------------------
# Chatbot Reply Function
# -----------------------------
def chatbot_reply(user_text: str, threshold: float = 0.4) -> str:
    query_emb = model.encode(user_text, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_emb, pattern_embeddings)[0]

    best_idx = int(torch.argmax(sims))
    best_score = float(sims[best_idx])
    best_tag = labels[best_idx]

    if best_score < threshold:
        return "🤔 Sorry, I’m not sure I understand. Could you rephrase?"

    responses = intents[best_tag]["responses"]
    return random.choice(responses) if responses else f"(No response defined for {best_tag})"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 Mental Health FAQ Chatbot (BERT)")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = [("Bot", "Hello! I’m your university assistant. How can I help you today?")]

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here:", "")
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.history.append(("User", user_input))
        bot_reply = chatbot_reply(user_input)
        st.session_state.history.append(("Bot", bot_reply))

# Display chat messages
chat_container = st.container()
with chat_container:
    for speaker, msg in st.session_state.history:
        if speaker == "User":
            st.markdown(
                f"<div style='text-align:right; background-color:#DCF8C6; padding:8px; border-radius:10px; margin:5px;'>"
                f"<b>{speaker}:</b> {msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; background-color:#ECECEC; padding:8px; border-radius:10px; margin:5px;'>"
                f"<b>{speaker}:</b> {msg}</div>",
                unsafe_allow_html=True
            )

# Auto-scroll trick
st.markdown("<div id='scroll-bottom'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <script>
        var scrollDiv = document.getElementById('scroll-bottom');
        if (scrollDiv) { scrollDiv.scrollIntoView({behavior: 'smooth'}); }
    </script>
    """,
    unsafe_allow_html=True
)
