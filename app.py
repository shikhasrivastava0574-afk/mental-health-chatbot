import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from transformers import pipeline


# -------------------- CONFIG --------------------
os.environ["GROQ_API_KEY"] = st.secrets["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"]

# -------------------- LOAD DATA --------------------
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


# -------------------- CRISIS --------------------
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


def detect_emotion(text):
    result = emotion_classifier(text)[0][0]
    return result["label"]


# -------------------- CHAT FUNCTION --------------------
if "history" not in st.session_state:
    st.session_state.history = []


def ask_question(question):

    if detect_crisis(question):
        return crisis_response()

    emotion = detect_emotion(question)

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a supportive mental health assistant.

    Emotion detected: {emotion}

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return f"(Emotion: {emotion})\n{response.content}"


# -------------------- UI --------------------
st.title("🌸 Mental Health AI Assistant")
st.write("You are not alone 💛")

user_input = st.text_input("How are you feeling today?")

if user_input:
    answer = ask_question(user_input)

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))


for role, text in st.session_state.history:
    st.write(f"**{role}:** {text}")

st.warning(
    "⚠️ This chatbot is not a medical professional. "
    "If you are in crisis, contact a licensed professional."
)