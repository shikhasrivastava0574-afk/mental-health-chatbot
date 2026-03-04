import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(page_title="Mental Health AI Assistant", layout="wide")


# ---------------- UI STYLE ---------------- #

st.markdown(
    """
    <style>

    .main {
        background-color: #e6f2f2;
    }

    .user-msg {
        background-color: #7f8c8d;
        padding: 15px;
        border-radius: 12px;
        color: black;
        font-size: 18px;
        margin-bottom: 10px;
    }

    .bot-msg {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 12px;
        color: black;
        font-size: 18px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- TITLE ---------------- #

st.markdown("# 🌿 Mental Health AI Assistant")
st.markdown("### You are not alone 💙")

# ---------------- LOAD PDFS ---------------- #

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

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-70b-8192"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- USER INPUT ---------------- #

query = st.chat_input("How are you feeling today?")


# ---------------- RESPONSE ---------------- #

if query:

    st.session_state.messages.append(("user", query))

    response = qa.run(query)

    st.session_state.messages.append(("bot", response))


# ---------------- DISPLAY CHAT ---------------- #

for role, message in st.session_state.messages:

    if role == "user":
        st.markdown(
            f'<div class="user-msg">🙂 {message}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="bot-msg">🤖 {message}</div>',
            unsafe_allow_html=True
        )


# ---------------- SAFETY MESSAGE ---------------- #

st.markdown(
    """
    ⚠️ **This chatbot is not a medical professional.  
    If you are in crisis, please contact a licensed therapist or mental health professional.**
    """
)
