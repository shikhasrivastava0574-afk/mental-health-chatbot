import streamlit as st
import os
import pandas as pd
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from transformers import pipeline


# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="Mental Health AI Assistant",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Mental Health AI Assistant")
st.write("You are not alone 💙")


# ---------------- EMOTION MODEL ---------------- #

@st.cache_resource
def load_emotion_model():

    model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

    return model


emotion_model = load_emotion_model()


# ---------------- LOAD PDF KNOWLEDGE ---------------- #

@st.cache_resource
def load_vectorstore():

    pdf_folder = "data/pdfs"

    documents = []

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------- LLM ---------------- #

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0.3
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood" not in st.session_state:
    st.session_state.mood = []


# ---------------- CHAT INPUT ---------------- #

query = st.chat_input("How are you feeling today?")


# ---------------- GENERATE RESPONSE ---------------- #

if query:

    st.session_state.messages.append(("user", query))

    # emotion detection
    emotion = emotion_model(query)[0]["label"]

    # rag response
    response = qa.run(query)

    st.session_state.messages.append(("assistant", response))

    st.session_state.mood.append({
        "time": datetime.now(),
        "emotion": emotion
    })


# ---------------- DISPLAY CHAT ---------------- #

for role, message in st.session_state.messages:

    with st.chat_message(role):
        st.write(message)


# ---------------- MOOD DASHBOARD ---------------- #

st.sidebar.title("📊 Mood Tracker")

if st.session_state.mood:

    df = pd.DataFrame(st.session_state.mood)

    mood_counts = df["emotion"].value_counts()

    st.sidebar.bar_chart(mood_counts)


# ---------------- FOOTER ---------------- #

st.markdown("""
---
⚠️ This chatbot is not a medical professional.  
If you are experiencing severe distress, please contact a licensed therapist or helpline.
""")
