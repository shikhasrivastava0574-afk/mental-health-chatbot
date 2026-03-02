import streamlit as st
import os
import pandas as pd
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mental Health Assistant",
    page_icon="🌿",
    layout="wide"
)


# ---------------- THEME ----------------
st.markdown("""
<style>

/* Background */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #cfe9f1, #e6f4f1) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f4fbfd !important;
}

/* User bubble */
.user-bubble {
    background-color: #daf1ff;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
}

/* Bot bubble */
.bot-bubble {
    background-color: #ffffff;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 8px 0;
    width: fit-content;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

/* Input */
.stTextInput input {
    background: white !important;
    color: black !important;
    border-radius: 12px;
    border: 1px solid #9ed0e6;
}

/* Label */
label {
    color: black !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ---------------- API KEY ----------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# ---------------- VECTOR DB ----------------
@st.cache_resource
def load_vectorstore():

    loader = PyPDFLoader("Managing-Stress-Social_Media.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------- MODELS ----------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood" not in st.session_state:
    st.session_state.mood = []


# ---------------- FUNCTIONS ----------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate_to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)


def detect_emotion(text):

    distress_words = ["not good", "sad", "low", "depressed", "stress"]

    if any(word in text.lower() for word in distress_words):
        return "sadness"

    result = emotion_classifier(text)[0][0]
    return result["label"]


def ask_question(question):

    lang = detect_language(question)

    if lang != "en":
        question_en = translate_to_english(question)
    else:
        question_en = question

    emotion = detect_emotion(question_en)

    docs = retriever.invoke(question_en)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a supportive mental health assistant.

    Emotion detected: {emotion}

    Context:
    {context}

    Question:
    {question_en}
    """

    response = llm.invoke(prompt)

    return emotion, response.content


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("💬 Chat History")

    if st.button("🆕 New Chat"):
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.write("🧑", msg["content"][:30])
        else:
            st.write("🤖", msg["content"][:30])

    st.divider()

    # Mood Tracker
    st.subheader("📊 Mood Tracker")

    if st.session_state.mood:
        df = pd.DataFrame(st.session_state.mood)
        st.bar_chart(df["emotion"].value_counts())

    st.divider()

    # Therapist Finder
    st.subheader("👩‍⚕️ Therapist Finder")

    city = st.selectbox(
        "City",
        ["Delhi", "Mumbai", "Bangalore", "Jaipur", "Online"]
    )

    if st.button("Find"):

        therapists = {
            "Delhi": ["Mind Care Clinic", "Serenity Mental Health"],
            "Mumbai": ["Hope Therapy Center"],
            "Bangalore": ["Inner Peace Clinic"],
            "Jaipur": ["Calm Minds Jaipur"],
            "Online": ["BetterHelp", "YourDOST", "MindPeers"]
        }

        for t in therapists[city]:
            st.write("•", t)


# ---------------- MAIN CHAT ----------------
st.title("🌿 Mental Health AI Assistant")
st.caption("You are not alone 💙")


# Display messages
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble"><b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-bubble"><b>Bot:</b> {msg["content"]}</div>',
            unsafe_allow_html=True
        )


# ---------------- INPUT ----------------
user_input = st.text_input("How are you feeling today?")

if user_input:

    emotion, answer = ask_question(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.messages.append({
        "role": "bot",
        "content": answer
    })

    st.session_state.mood.append({
        "time": datetime.now(),
        "emotion": emotion
    })

    st.rerun()


# ---------------- FOOTER ----------------
st.markdown("""
---
⚠️ This chatbot is not a medical professional.
If you are in crisis, please contact a licensed therapist.
""")
