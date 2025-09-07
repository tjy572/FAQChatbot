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
    ("Keyword Matching", "Chatbot"): "http://localhost:9501/keyword_matching",
    ("Keyword Matching", "Evaluation"): "http://localhost:9503/evaluationKeywordMatching",
    ("TF-IDF", "Chatbot"): "http://localhost:9506/tfidf_chatbot",
    ("TF-IDF", "Evaluation"): "http://localhost:9507/eval_tfidf",
    ("BERT", "Chatbot"): "http://localhost:9504/bertChatbot",
    ("BERT", "Evaluation"): "http://localhost:9505/evaluationBert",
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
