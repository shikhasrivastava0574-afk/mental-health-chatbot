import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Mental Health Assistant 🌿",
    page_icon="💙",
    layout="centered"
)


# -------------------- THERAPY CALM THEME --------------------
st.markdown("""
<style>

/* Calm gradient background */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #cfe9f1 0%, #e6f4f1 50%, #d8eefe 100%) !important;
}

/* Soft floating light effect */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image: radial-gradient(#ffffff40 2px, transparent 2px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* Remove header */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Main container */
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

/* Input box */
.stTextInput > div > div > input {
    background-color: white !important;
    color: #333 !important;
    border-radius: 20px;
    border: 1.5px solid #9ed0e6;
    padding: 14px;
    font-size: 16px;
}

/* Chat card (glass effect) */
.chat-box {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-top: 20px;
    color: #1f2d3d;
    font-size: 17px;
    line-height: 1.7;
}

/* Title */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #2b6f89;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #4a6a7a;
    margin-bottom: 25px;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)


# -------------------- CONFIG --------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# -------------------- LOAD VECTOR DB --------------------
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

    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -------------------- MODELS --------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


# -------------------- LANGUAGE --------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def detect_language_type(text):

    hinglish_words = ["mujhe", "tum", "nahi", "hai", "ho", "lg"]

    if any(word in text.lower() for word in hinglish_words):
        return "hinglish"

    lang = detect_language(text)

    if lang == "hi":
        return "hindi"

    return "english"


def translate_to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)


def translate_to_hindi(text):
    return GoogleTranslator(source="en", target="hi").translate(text)


# -------------------- EMOTION --------------------
def detect_emotion(text):

    distress_words = [
        "not good", "sad", "bad", "low",
        "depressed", "anxious", "stress"
    ]

    if any(word in text.lower() for word in distress_words):
        return "sadness"

    result = emotion_classifier(text)[0][0]
    emotion = result["label"]

    if emotion == "surprise":
        emotion = "confusion"

    return emotion


# -------------------- CHAT --------------------
def ask_question(question):

    lang_type = detect_language_type(question)

    if lang_type in ["hindi", "hinglish"]:
        question_en = translate_to_english(question)
    else:
        question_en = question

    emotion = detect_emotion(question_en)

    docs = retriever.invoke(question_en)
    context = "\n\n".join([doc.page_content for doc in docs])

    if lang_type == "hindi":
        language_instruction = "Reply in Hindi."

    elif lang_type == "hinglish":
        language_instruction = "Reply in natural Hinglish."

    else:
        language_instruction = "Reply in English."

    prompt = f"""
    You are a calm, supportive mental health assistant.

    Emotion detected: {emotion}

    Context:
    {context}

    Question:
    {question_en}

    {language_instruction}
    """

    response = llm.invoke(prompt)
    answer = response.content

    return emotion, answer


# -------------------- UI --------------------
st.markdown('<div class="title">🌿 Mental Health AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">You are not alone 💙 | Hindi • Hinglish • English</div>', unsafe_allow_html=True)

user_input = st.text_input("How are you feeling today? 🌱")

if user_input:

    emotion, answer = ask_question(user_input)

    st.markdown(f"""
    <div class="chat-box">
    🧠 <b>Emotion:</b> {emotion} <br><br>
    🤖 <b>Assistant:</b><br>
    {answer}
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center; margin-top:30px; color:#5a6d75;">
⚠️ This chatbot is not a medical professional.
If you are in crisis, contact a licensed professional.
</div>
""", unsafe_allow_html=True)
