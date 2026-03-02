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

from streamlit_mic_recorder import mic_recorder


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


# -------------------- LANGUAGE FUNCTIONS --------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def detect_language_type(text):
    lang = detect_language(text)

    hinglish_words = [
        "mujhe", "tum", "kya", "kyun", "nahi", "hai",
        "ho", "raha", "kar", "mera", "apna", "kaise",
        "bahut", "dil", "mann", "lag", "accha"
    ]

    text_lower = text.lower()

    if any(word in text_lower for word in hinglish_words):
        return "hinglish"

    if lang == "hi":
        return "hindi"

    return "english"


def translate_to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)


def translate_to_hindi(text):
    return GoogleTranslator(source="en", target="hi").translate(text)


# -------------------- CRISIS DETECTION --------------------
crisis_keywords = [
    "suicide", "kill myself", "want to die",
    "end my life", "self harm", "hurt myself"
]


def detect_crisis(text):
    text = text.lower()
    return any(word in text for word in crisis_keywords)


def crisis_response():
    return """
🚨 You are not alone.

If you are in immediate danger please contact:

📞 Kiran Mental Health Helpline: 1800-599-0019  
📞 AASRA: +91-9820466726  

Please reach out to someone you trust 💛
"""


# -------------------- EMOTION DETECTION --------------------
def detect_emotion(text):
    result = emotion_classifier(text)[0][0]
    return result["label"]


# -------------------- CHAT FUNCTION --------------------
def ask_question(question):

    lang_type = detect_language_type(question)

    # Translate to English for processing
    if lang_type in ["hindi", "hinglish"]:
        question_en = translate_to_english(question)
    else:
        question_en = question

    # Crisis detection
    if detect_crisis(question_en):
        response = crisis_response()

        if lang_type == "hindi":
            return translate_to_hindi(response)

        if lang_type == "hinglish":
            prompt = f"Convert this into natural Hinglish (Roman Hindi): {response}"
            return llm.invoke(prompt).content

        return response

    # Emotion detection
    emotion = detect_emotion(question_en)

    # Retrieval
    docs = retriever.invoke(question_en)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Language instruction
    if lang_type == "hindi":
        language_instruction = "Reply in Hindi."

    elif lang_type == "hinglish":
        language_instruction = "Reply in natural Hinglish (Roman Hindi, casual Indian style)."

    else:
        language_instruction = "Reply in English."

    prompt = f"""
    You are a supportive mental health assistant.

    Emotion detected: {emotion}

    Context:
    {context}

    Question:
    {question_en}

    {language_instruction}

    Give a caring and supportive answer.
    """

    response = llm.invoke(prompt)
    answer = response.content

    return f"(Emotion: {emotion})\n{answer}"


# -------------------- UI --------------------
st.title("🌸 Mental Health AI Assistant")
st.write("You are not alone 💛")
st.write("Hindi / Hinglish / Voice supported 🇮🇳🎤")

st.write("🎤 Speak instead of typing")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True
)

user_input = ""

if audio:

    if "text" in audio and audio["text"]:
        text = audio["text"]
    else:
        text = ""

    st.write("You said:", text)
    user_input = text

else:
    user_input = st.text_input("How are you feeling today?")


if user_input:
    answer = ask_question(user_input)
    st.write("Bot:", answer)


st.warning(
    "⚠️ This chatbot is not a medical professional. "
    "If you are in crisis, contact a licensed professional."
)
