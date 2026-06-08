# 🌿 ZenFlow — AI Mental Companion

ZenFlow is an innovative, highly interactive mental health companion built with Streamlit and LangChain. It combines a RAG-powered chatbot with sentiment/emotion tracking, mindfulness exercises, soundscapes, and personal reflection tools to create a supportive digital space for emotional well-being.

---

## ✨ Features

ZenFlow provides a suite of interactive tools to help you manage stress, reflect, and stay grounded:

1. **🌿 AI Care Space (Chat Assistant):** 
   - A supportive conversational interface.
   - Detects the emotional tone of your messages in real-time (Joy, Sadness, Anger, Fear, Surprise, Love, Neutral) using a local Hugging Face DistilRoBERTa classification model.
   - Shows active emotion badges and dynamically adjusts the theme's glow based on your feelings.
   - Leverages RAG (Retrieval-Augmented Generation) to search uploaded self-help documents for therapeutic advice.

2. **📊 Mood Analytics & Tracker:**
   - Log your daily mood manually or track it automatically from chat sessions.
   - View visual trends and distributions of your emotional state over time via charts.

3. **🧘 Mindful Exercises:**
   - **🌀 Box Breathing:** A CSS-animated guide that expands and contracts in sync with a 16-second breathing rhythm (4s inhale, 4s hold, 4s exhale, 4s hold) to help calm your nervous system.
   - **🎯 5-4-3-2-1 Grounding Game:** A step-by-step interactive wizard designed to ease acute anxiety by engaging all five senses.

4. **📖 Daily Reflection Journal:**
   - A private digital diary to write entries.
   - ZenFlow automatically analyzes the emotional sentiment of your writing and provides customized positive affirmations.
   - Keeps a history of your past entries in an organized accordion view.

5. **🎯 Daily Self-Care Tracker:**
   - Check off daily wellness habits (sleep, water intake, exercise, hydration).
   - Real-time progress bar with celebration animations when goals are fully met.

6. **🎵 Calming Soundscapes:**
   - Background audio player playing summer rain, ocean waves, forest ambiance, or soft piano lofi directly in the browser.

7. **🚨 Crisis Center:**
   - A clean card directory of mental health support hotlines (US, Canada, UK) for quick access.

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit, Custom CSS Injection, HTML5 Audio, CSS Keyframe Animations
- **LLM/RAG Orchestration:** LangChain, LangChain Classic
- **Embeddings & Vectorstore:** `sentence-transformers/all-MiniLM-L6-v2`, FAISS (Facebook AI Similarity Search)
- **Local Emotion Classifier:** `j-hartmann/emotion-english-distilroberta-base` (DistilRoBERTa model run locally via `transformers`)
- **LLM Engine:** ChatGroq (`llama3-70b-8192`)
- **Data Handling:** Pandas, Python datetime

---

## 🚀 Installation & Local Setup

### Prerequisites

Make sure you have Python 3.10+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/shikhasrivastava0574-afk/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Knowledge PDF (Optional)
Place any mental health support or mindfulness PDFs inside the `data/pdfs/` directory. The application will automatically split and load them into the FAISS vector database to customize the chatbot's knowledge base.

### 4. Configure API Keys (Optional but Recommended)
For conversational AI chat, set up a free API key from the [Groq Console](https://console.groq.com/). You can set it in your environment:
```bash
export GROQ_API_KEY="your-groq-api-key"
```
*Alternatively, you can paste the key directly inside the ZenFlow sidebar when the app is running.*

### 5. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to start using ZenFlow!

---

## ☁️ Deployment on Hugging Face Spaces

ZenFlow is fully compatible with Hugging Face Spaces out of the box.

1. Create a new Streamlit Space on [Hugging Face](https://huggingface.co/spaces).
2. Upload all the repository files (including the `requirements.txt` and `app.py`).
3. Set your `GROQ_API_KEY` under the **Repository Secrets** in Space Settings.
4. Hugging Face will automatically install the requirements and deploy the app!

---

## ⚠️ Medical Disclaimer

ZenFlow is an AI-powered conversational support companion. It is designed to provide mindfulness exercises, general coping tips, and RAG-based context from self-help documents. It does not provide medical diagnostics, clinical therapy, or medical prescriptions. If you are experiencing a mental health emergency, please seek immediate help from a healthcare provider or a licensed counselor.
