# keyword_matching.py
import streamlit as st
import csv, json, re, random, unicodedata

# -------------------------
# Load and clean datasets
# -------------------------
def clean_text_dataset(text):
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'"}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii").strip()

qa_pairs = []

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

# -------------------------
# Keyword matching logic
# -------------------------
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
st.set_page_config(page_title="Keyword Matching Chatbot", layout="wide")
st.title("🤖 Mental Health Chatbot - Keyword Matching")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = [("Bot", "Hello! I'm your assistant. How can I help you today?")]

#input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here:", "")
    submitted = st.form_submit_button("Send")
    
    if submitted and user_input.strip():
        st.session_state.history.append(("User", user_input))
        bot_reply, _ = best_match(user_input)
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

