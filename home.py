import streamlit as st

# Page config
st.set_page_config(
    page_title="Mental Health Chatbot Hub",
    page_icon="🤖",
    layout="centered"
)

# Title & instructions
st.title("🤖 Mental Health Chatbot Hub")
st.write("Select an algorithm and mode to continue:")

# Dropdowns
algo = st.selectbox("Select Algorithm", ["", "Keyword Matching", "TF-IDF", "BERT"])
mode = st.selectbox("Select Mode", ["", "Chatbot", "Evaluation"])

# Mapping choices to URLs
url_mapping = {
    ("Keyword Matching", "Chatbot"): "https://faqchatbot-keyword-matching.streamlit.app/",
    ("Keyword Matching", "Evaluation"): "https://faqchatbot-eval-key-matching.streamlit.app/",
    ("TF-IDF", "Chatbot"): "https://faqchatbot-tfdif-chatbot.streamlit.app/",
    ("TF-IDF", "Evaluation"): "https://faqchatbot-tfdif-eval.streamlit.app/",
    ("BERT", "Chatbot"): "https://faqchatbot-bert-chatbot.streamlit.app/",
    ("BERT", "Evaluation"): "https://faqchatbot-eval-bert.streamlit.app/",
}

# Launch button
if st.button("🚀 Launch"):
    if algo and mode:
        url = url_mapping.get((algo, mode))
        if url:
            st.success(f"Launching {algo} - {mode}...")
            st.markdown(
                f"[👉 Click here to open {algo} {mode}]({url})",
                unsafe_allow_html=True
            )
        else:
            st.error("Invalid selection.")
    else:
        st.warning("⚠️ Please select both algorithm and mode before launching.")
