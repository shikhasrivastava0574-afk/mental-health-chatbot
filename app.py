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


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Mental Health Assistant",
    page_icon="🌿",
    layout="wide"
)


# -------------------------------------------------
# CSS (FONT FIX)
# -------------------------------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#cfe9f1,#e6f4f1);
}

section[data-testid="stSidebar"]{
background:#f4fbfd;
}

body, p, span, div, label, h1, h2, h3{
color:black !important;
}

[data-testid="stChatMessageContent"]{
color:black !important;
}

.stTextInput input{
background:white !important;
color:black !important;
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# API KEY
# -------------------------------------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# -------------------------------------------------
# LOAD VECTOR DATABASE
# -------------------------------------------------

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
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


# -------------------------------------------------
# MODELS
# -------------------------------------------------

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

emotion_model = pipeline(
"text-classification",
model="j-hartmann/emotion-english-distilroberta-base"
)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood" not in st.session_state:
    st.session_state.mood = []

if "last_message" not in st.session_state:
    st.session_state.last_message = ""


# -------------------------------------------------
# LANGUAGE FUNCTIONS
# -------------------------------------------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate_to_english(text):
    return GoogleTranslator(source="auto",target="en").translate(text)


# -------------------------------------------------
# EMOTION DETECTION (FIXED FOR HINGLISH)
# -------------------------------------------------

def detect_emotion(text):

    negative_words = [
        "not good",
        "sad",
        "low",
        "depressed",
        "stress",
        "accha nahi",
        "acha nahi",
        "bura lag raha",
        "mujhe acha nahi",
        "thik nahi"
    ]

    for word in negative_words:
        if word in text.lower():
            return "sadness"

    result = emotion_model(text)[0]

    return result["label"]


# -------------------------------------------------
# CRISIS DETECTION
# -------------------------------------------------

def crisis_detection(text):

    crisis_words = [
    "kill myself",
    "suicide",
    "want to die",
    "end my life"
    ]

    for word in crisis_words:
        if word in text.lower():
            return True

    return False


# -------------------------------------------------
# CHATBOT FUNCTION
# -------------------------------------------------

def ask_question(question):

    lang = detect_language(question)

    if lang!="en":
        question_en = translate_to_english(question)
    else:
        question_en = question

    emotion = detect_emotion(question_en)

    docs = retriever.invoke(question_en)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a compassionate mental health support assistant.

IMPORTANT RULES:
- If the user says they feel bad, sad, stressed, anxious, or "not good", respond with empathy.
- Never assume the user is happy if they say they are not feeling good.
- Do not hallucinate imaginary scenarios.
- Be calm and supportive.
- Ask gentle follow-up questions.

Detected Emotion: {emotion}

Context from mental health guide:
{context}

User message:
{question_en}

Respond in the same language as the user (Hindi, Hinglish, or English).
"""

    response = llm.invoke(prompt)

    return emotion,response.content


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.title("💬 Chat History")

    if st.button("🆕 New Chat"):
        st.session_state.messages=[]

    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.write("🙂",msg["content"][:40])
        else:
            st.write("🤖",msg["content"][:40])

    st.divider()

    st.subheader("📊 Mood Tracker")

    if st.session_state.mood:
        df = pd.DataFrame(st.session_state.mood)
        st.bar_chart(df["emotion"].value_counts())

    st.divider()

    st.subheader("👩‍⚕️ Therapist Finder")

    city = st.selectbox(
    "City",
    ["Delhi","Mumbai","Bangalore","Jaipur","Online"]
    )

    if st.button("Find Therapist"):

        therapists={
        "Delhi":["Mind Care Clinic","Serenity Therapy"],
        "Mumbai":["Hope Therapy Center"],
        "Bangalore":["Inner Peace Clinic"],
        "Jaipur":["Calm Minds"],
        "Online":["BetterHelp","YourDOST","MindPeers"]
        }

        for t in therapists[city]:
            st.write("•",t)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🌿 Mental Health AI Assistant")
st.write("You are not alone 💙")


# -------------------------------------------------
# DISPLAY CHAT
# -------------------------------------------------

for msg in st.session_state.messages:

    if msg["role"]=="user":

        with st.chat_message("user"):
            st.write(msg["content"])

    else:

        with st.chat_message("assistant"):
            st.write(msg["content"])


# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------

user_input = st.chat_input("How are you feeling today?")

if user_input and st.session_state.last_message != user_input:

    st.session_state.last_message = user_input

    st.session_state.messages.append({
    "role":"user",
    "content":user_input
    })

    if crisis_detection(user_input):

        response = """
I'm really sorry you're feeling this way.  
You are not alone.

Please consider reaching out to someone immediately.

India Suicide Helpline:
9152987821

You deserve support and care.
"""

        emotion="crisis"

    else:

        emotion,response = ask_question(user_input)

    st.session_state.messages.append({
    "role":"assistant",
    "content":response
    })

    st.session_state.mood.append({
    "time":datetime.now(),
    "emotion":emotion
    })


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("""
---
⚠️ This chatbot is not a medical professional.  
If you are in crisis please contact a licensed therapist.
""")
