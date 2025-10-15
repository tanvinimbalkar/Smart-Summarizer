import streamlit as st
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

# Silence warnings
set_verbosity_error()

# Page Config
st.set_page_config(page_title="Text Summarizer & QnA", layout="centered")

# Custom Style
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #E8F0FE !important;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        [data-testid="stSidebar"] {
            background-color: #E8F0FE !important;
        }
        .title {
            text-align: center;
            font-size: 2.2rem;
            color: #1E3A8A;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .subheader {
            text-align: center;
            color: #475569;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .stTextArea textarea {
            border-radius: 10px !important;
        }
        .stButton>button {
            background-color: #1E40AF;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #2563EB;
        }
        .output-box {
            background-color: #C7D2FE;
            padding: 1rem;
            border-radius: 10px;
            color: #1E3A8A;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
st.markdown("<div class='title'>🧠 Smart Text Summarizer & QnA Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Summarize long text and ask questions about it easily!</div>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    refiner = pipeline("summarization", model="facebook/bart-large")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, refiner, qa_pipeline

summarizer, refiner, qa_pipeline = load_pipelines()

# Summarization
text_to_summarize = st.text_area("Enter text to summarize:", height=200)
length = st.radio("Select summary length:", ["short", "medium", "long"], horizontal=True)

# Define summary length dynamically
length_settings = {
    "short": {"min_length": 30, "max_length": 80},
    "medium": {"min_length": 80, "max_length": 160},
    "long": {"min_length": 160, "max_length": 300},
}

if st.button("Summarize"):
    if text_to_summarize.strip():
        with st.spinner("Generating summary..."):
            try:
                # Apply variable summary lengths
                params = length_settings[length]
                raw_summary = summarizer(text_to_summarize, **params)
                summary = raw_summary[0]["summary_text"]

                # Optionally refine
                refined_summary = refiner(summary, min_length=20, max_length=150)[0]["summary_text"]

                st.session_state["summary"] = refined_summary

                st.markdown("### 🔹 Generated Summary:")
                st.markdown(f"<div class='output-box'>{refined_summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text to summarize.")

# Q&A Section
if "summary" in st.session_state:
    st.markdown("---")
    st.subheader("❓ Ask a question about the summary:")
    question = st.text_input("Type your question here:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Finding answer..."):
                qa_result = qa_pipeline(question=question, context=st.session_state["summary"])
            st.markdown("### 🔹 Answer:")
            st.markdown(f"<div class='output-box'>{qa_result['answer']}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a question.")
