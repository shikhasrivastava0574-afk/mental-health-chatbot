import streamlit as st
import os
import pandas as pd
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA

from transformers import pipeline

# ---------------- CONFIGURATION & PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="🌿 ZenFlow - AI Mental Companion",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE INITIALIZATION ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood_history" not in st.session_state:
    st.session_state.mood_history = [
        {"time": datetime.now(), "emotion": "neutral", "notes": "Started using ZenFlow! 💙"}
    ]

if "journal_entries" not in st.session_state:
    st.session_state.journal_entries = []

if "checklist" not in st.session_state:
    st.session_state.checklist = {
        "💧 Drink 8 glasses of water": False,
        "🚶 Take a 15-minute walk": False,
        "🧘 Do a breathing exercise": False,
        "📖 Write in my journal": False,
        "🛌 Sleep 8 hours": False,
        "🍎 Eat a healthy meal": False
    }

if "grounding_step" not in st.session_state:
    st.session_state.grounding_step = 0

if "grounding_answers" not in st.session_state:
    st.session_state.grounding_answers = {}

# ---------------- MODELS & DATA LOADING (CACHED) ---------------- #

@st.cache_resource
def load_emotion_model():
    try:
        model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        return model
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None

@st.cache_resource
def load_vectorstore():
    pdf_folder = "data/pdfs"
    documents = []
    
    # Check if folder exists
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        
    # Check for PDFs
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        # Create a fallback document to prevent FAISS error if no PDFs are uploaded yet
        from langchain.docstore.document import Document
        fallback_doc = Document(
            page_content="ZenFlow is a supportive AI companion. For stress relief, focus on deep breathing, mindfulness, daily hydration, physical movement, and regular sleep.",
            metadata={"source": "system_default"}
        )
        docs = [fallback_doc]
    else:
        for file in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(pdf_folder, file))
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Error loading PDF {file}: {e}")
                
        if not documents:
            from langchain.docstore.document import Document
            documents = [Document(page_content="Mental health support information.", metadata={"source": "system"})]
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        docs = splitter.split_documents(documents)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vectorstore: {e}")
        return None

# Load models
emotion_model = load_emotion_model()
vectorstore = load_vectorstore()

# Setup retriever if vectorstore is loaded
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

def get_detected_emotion(text):
    if not text or not text.strip():
        return "neutral"
    text_lower = text.lower().strip()
    
    # 1. Simple heuristic overrides for negations and common triggers
    if any(phrase in text_lower for phrase in ["not good", "not feeling good", "feeling bad", "not okay", "not ok", "not fine", "sad", "depressed", "unhappy"]):
        return "sadness"
    if any(phrase in text_lower for phrase in ["anxious", "scared", "afraid", "panic", "worried", "nervous"]):
        return "fear"
    if any(phrase in text_lower for phrase in ["angry", "mad", "annoyed", "frustrated", "pissed"]):
        return "anger"
    if any(phrase in text_lower for phrase in ["happy", "great", "awesome", "excellent", "wonderful", "joy"]):
        if not any(neg in text_lower for neg in ["not", "never", "don't", "dont", "no"]):
            return "joy"
            
    # 2. Fallback to ML Model
    if not emotion_model:
        return "neutral"
    try:
        predictions = emotion_model(text)
        if not predictions:
            return "neutral"
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                predictions = predictions[0]
            best_prediction = max(predictions, key=lambda x: x.get("score", 0))
            return best_prediction.get("label", "neutral")
    except Exception:
        pass
    return "neutral"

# ---------------- STYLING SYSTEM (INJECTED CSS) ---------------- #

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Fonts & Style override */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Modern Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .glass-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        text-align: center;
    }
    
    /* Custom Chat Interface */
    .chat-bubble-container {
        display: flex;
        flex-direction: column;
        margin-bottom: 16px;
    }
    
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 18px;
        max-width: 80%;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        display: inline-block;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #4F46E5, #3B82F6);
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.08);
        color: #E5E7EB;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .chat-meta {
        font-size: 0.75rem;
        margin-top: 4px;
        color: #9CA3AF;
        align-self: flex-start;
    }
    
    .chat-meta-user {
        font-size: 0.75rem;
        margin-top: 4px;
        color: #A5F3FC;
        align-self: flex-end;
    }
    
    /* Emotion Badge styling */
    .emotion-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
        color: white;
        text-transform: capitalize;
        gap: 5px;
    }
    
    .emotion-joy { background-color: #10B981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.3); }
    .emotion-sadness { background-color: #3B82F6; box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
    .emotion-anger { background-color: #EF4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
    .emotion-fear { background-color: #F59E0B; box-shadow: 0 0 10px rgba(245, 158, 11, 0.3); }
    .emotion-surprise { background-color: #8B5CF6; box-shadow: 0 0 10px rgba(139, 92, 246, 0.3); }
    .emotion-love { background-color: #EC4899; box-shadow: 0 0 10px rgba(236, 72, 153, 0.3); }
    .emotion-neutral { background-color: #6B7280; box-shadow: 0 0 10px rgba(107, 114, 128, 0.3); }
    .emotion-disgust { background-color: #7C3AED; box-shadow: 0 0 10px rgba(124, 58, 237, 0.3); }
    
    /* Breathing Circle Visualizer */
    .breathing-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 320px;
        margin: 30px 0;
    }
    
    .breathing-circle {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.85) 0%, rgba(59, 130, 246, 0.4) 100%);
        box-shadow: 0 0 35px rgba(99, 102, 241, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        animation: breathe-cycle 16s infinite ease-in-out;
        position: relative;
    }
    
    .breathing-circle::after {
        content: "Hold";
        animation: breathing-text 16s infinite ease-in-out;
        position: absolute;
    }
    
    @keyframes breathe-cycle {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
            background: radial-gradient(circle, rgba(99, 102, 241, 0.8) 0%, rgba(59, 130, 246, 0.4) 100%);
        }
        25% {
            transform: scale(1.6);
            box-shadow: 0 0 60px rgba(99, 102, 241, 0.9);
            background: radial-gradient(circle, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.6) 100%);
        }
        50% {
            transform: scale(1.6);
            box-shadow: 0 0 60px rgba(139, 92, 246, 0.9);
            background: radial-gradient(circle, rgba(139, 92, 246, 0.9) 0%, rgba(99, 102, 241, 0.6) 100%);
        }
        75% {
            transform: scale(1);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
            background: radial-gradient(circle, rgba(99, 102, 241, 0.8) 0%, rgba(59, 130, 246, 0.4) 100%);
        }
    }
    
    @keyframes breathing-text {
        0%, 100% { content: "Hold"; }
        5%, 20% { content: "Inhale"; }
        25%, 45% { content: "Hold"; }
        50%, 70% { content: "Exhale"; }
        75%, 95% { content: "Hold"; }
    }
    
    /* Interactive Card Hover effects */
    .hover-card {
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .hover-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Custom footer */
    .disclaimer-box {
        border-left: 4px solid #F59E0B;
        background-color: rgba(245, 158, 11, 0.08);
        padding: 15px;
        border-radius: 0 12px 12px 0;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------- API KEY CONFIGURATION ---------------- #

# Check environment variable first
groq_api_key = os.getenv("GROQ_API_KEY")

# Create sidebar setup for Groq Key if not in env
st.sidebar.markdown("### 🔑 API Configuration")
if not groq_api_key:
    user_entered_key = st.sidebar.text_input("Enter GROQ API Key", type="password", help="Get a free key from console.groq.com")
    if user_entered_key:
        groq_api_key = user_entered_key
        st.sidebar.success("API Key applied!")
    else:
        st.sidebar.warning("GROQ Key missing. Chat bot fallback active.")
else:
    st.sidebar.success("GROQ Key loaded from environment")

# Initialize LLM & Chain
qa_chain = None
if groq_api_key:
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.3
        )
        if retriever:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )
    except Exception as e:
        st.sidebar.error(f"Error initializing LLM: {e}")

# ---------------- NAVIGATION SIDEBAR ---------------- #

st.sidebar.title("🌿 ZenFlow Menu")
page = st.sidebar.radio(
    "Choose a space:",
    [
        "🌿 AI Care Space", 
        "📊 Mood Analytics", 
        "🧘 Breathing & Grounding", 
        "📖 Daily Reflection Journal", 
        "🎯 Self-Care Tracker",
        "🎵 Calming Soundscapes",
        "🚨 Crisis & Resource Center"
    ]
)

# helper emojis for emotion classification
emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😟",
    "surprise": "😮",
    "love": "🥰",
    "neutral": "😐"
}

# ---------------- PAGE 1: AI CARE SPACE (CHAT) ---------------- #

if page == "🌿 AI Care Space":
    
    st.markdown('<div class="glass-header"><h1>🌿 AI Care Space</h1><p>I am here to listen and support you. Share whatever is on your mind. 💙</p></div>', unsafe_allow_html=True)
    
    # Show active emotion glow in the interface based on last user response
    current_mood_class = "emotion-neutral"
    current_mood_text = "Neutral"
    current_emoji = "😐"
    
    user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
    if user_messages and "emotion" in user_messages[-1]:
        last_emotion = user_messages[-1]["emotion"]
        current_mood_class = f"emotion-{last_emotion}"
        current_mood_text = last_emotion
        current_emoji = emotion_emojis.get(last_emotion, "😐")
        
    st.markdown(f"""
    <div style="text-align: right; margin-bottom: 20px;">
        <span style="font-size: 0.9rem; color: #9CA3AF; margin-right: 8px;">Active Tone Sensitivity:</span>
        <span class="emotion-badge {current_mood_class}">{current_emoji} {current_mood_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Display message history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            emotion = msg.get("emotion", None)
            
            if role == "user":
                emoji = emotion_emojis.get(emotion, "😐") if emotion else ""
                badge_html = f'<div class="emotion-badge emotion-{emotion}">{emoji} {emotion}</div>' if emotion else ""
                
                st.markdown(f"""
                <div class="chat-bubble-container">
                    <div class="chat-bubble user-bubble">
                        {content}
                    </div>
                    {badge_html}
                    <div class="chat-meta-user">You</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble-container">
                    <div class="chat-bubble assistant-bubble">
                        {content}
                    </div>
                    <div class="chat-meta">ZenFlow Assistant</div>
                </div>
                """, unsafe_allow_html=True)
                
    # Chat Input
    query = st.chat_input("How are you feeling right now?")
    
    if query:
        # Detect emotion
        detected_emotion = get_detected_emotion(query)
                
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "emotion": detected_emotion
        })
        
        # Log to mood tracker
        st.session_state.mood_history.append({
            "time": datetime.now(),
            "emotion": detected_emotion,
            "notes": f"Detected in chat: '{query[:30]}...'"
        })
        
        # Generate assistant response
        with st.spinner("ZenFlow is reflecting..."):
            if qa_chain:
                try:
                    response = qa_chain.run(query)
                except Exception as e:
                    response = f"I apologize, but I encountered an error while processing your request: {e}. Please check your connection or try again."
            else:
                # Friendly fallback empathetic responses based on emotion
                fallback_responses = {
                    "joy": "I am so happy to hear that! It sounds like you are feeling joyful. Capturing and appreciating these moments of happiness is wonderful for your mental well-being. What is making you feel this way?",
                    "sadness": "I hear you, and I am so sorry you are feeling sad. Please know that it is completely okay to feel this way, and you are not alone. Be gentle with yourself today. Would you like to try our Box Breathing or Grounding exercise in the menu?",
                    "anger": "It sounds like you're carrying some anger or frustration, and that is completely valid. It can help to express these feelings. If you need a moment to release tension, perhaps we can do a brief Box Breathing session together.",
                    "fear": "It feels like you are experiencing some anxiety or fear right now. Take a slow, deep breath. You are safe in this moment. I highly recommend checking out our Grounding Technique (in the Breathing & Grounding page) to help bring you back to the present.",
                    "love": "What a beautiful feeling. Love and connection are so core to our resilience. Thank you for sharing that warm energy with me! Tell me more about it.",
                    "surprise": "Oh, that sounds like a surprising turn of events! How are you processing this change? I'm here if you want to talk it through.",
                    "neutral": "Thank you for sharing that with me. I'm here to listen. Tell me more about how you're feeling or what has been happening today."
                }
                
                # Check for Groq recommendation
                suggest_key = " (Note: You can add your GROQ API Key in the sidebar for full conversational AI therapy powered by your PDFs!)"
                response = fallback_responses.get(detected_emotion, fallback_responses["neutral"]) + suggest_key
                
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()

# ---------------- PAGE 2: MOOD ANALYTICS ---------------- #

elif page == "📊 Mood Analytics":
    st.markdown('<div class="glass-header"><h1>📊 Mood & Analytics Tracker</h1><p>Understand your emotional landscape over time. 💙</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card"><h3>📝 Log Current Mood</h3></div>', unsafe_allow_html=True)
        manual_emotion = st.selectbox(
            "How do you feel right now?",
            ["neutral", "joy", "sadness", "anger", "fear", "surprise", "love"]
        )
        manual_notes = st.text_input("Any notes? (e.g. 'Finished a big project', 'Restless')")
        
        if st.button("Log Mood", use_container_width=True):
            st.session_state.mood_history.append({
                "time": datetime.now(),
                "emotion": manual_emotion,
                "notes": manual_notes if manual_notes else "Manual Log"
            })
            st.success("Mood logged successfully!")
            st.rerun()
            
    with col2:
        st.markdown('<div class="glass-card"><h3>📈 Mood Distribution</h3></div>', unsafe_allow_html=True)
        if st.session_state.mood_history:
            df = pd.DataFrame(st.session_state.mood_history)
            
            # Format timestamp
            df['date'] = pd.to_datetime(df['time']).dt.strftime('%b %d, %H:%M')
            
            # Mood counts
            mood_counts = df["emotion"].value_counts().reset_index()
            mood_counts.columns = ['Emotion', 'Count']
            
            # Display simple bar chart
            st.bar_chart(data=mood_counts, x="Emotion", y="Count", use_container_width=True)
        else:
            st.info("No mood logs yet. Try talking to the AI or logging your mood above!")
            
    st.markdown('<div class="glass-card"><h3>📜 Mood History Log</h3></div>', unsafe_allow_html=True)
    if st.session_state.mood_history:
        history_df = pd.DataFrame(st.session_state.mood_history).sort_values(by="time", ascending=False)
        for _, row in history_df.iterrows():
            emoji = emotion_emojis.get(row['emotion'], "😐")
            time_str = pd.to_datetime(row['time']).strftime('%b %d, %Y - %I:%M %p')
            
            st.markdown(f"""
            <div style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem; margin-right: 8px;">{emoji}</span>
                    <strong style="text-transform: capitalize;">{row['emotion']}</strong>
                    <span style="color: #9CA3AF; margin-left: 15px; font-size: 0.9rem;">{row['notes']}</span>
                </div>
                <div style="color: #6B7280; font-size: 0.85rem;">{time_str}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- PAGE 3: BREATHING & GROUNDING ---------------- #

elif page == "🧘 Breathing & Grounding":
    st.markdown('<div class="glass-header"><h1>🧘 Mindful Exercises</h1><p>Take a moment to center yourself and return to the present. 💙</p></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🌀 Box Breathing", "🎯 5-4-3-2-1 Grounding Technique"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### What is Box Breathing?
            Box breathing (also known as four-square breathing) is a simple powerful relaxation technique that can help clear your mind, relax your body, and improve focus. 
            
            It is used by everyone from professional athletes to US Navy SEALs to manage stress and anxiety.
            
            #### Instructions:
            1. **Inhale** slowly through your nose for 4 seconds.
            2. **Hold** your breath in for 4 seconds.
            3. **Exhale** completely through your mouth for 4 seconds.
            4. **Hold** your empty lungs for 4 seconds.
            5. Repeat this cycle 4 times or until calm.
            """)
            
        with col2:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("### 🌀 Animated Rhythm Guide")
            st.markdown("Follow the expansion of the bubble to sync your breath.")
            
            # Breathing visualizer using raw HTML/CSS
            st.markdown("""
            <div class="breathing-wrapper">
                <div class="breathing-circle"></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    with tab2:
        st.markdown("""
        ### 🎯 The 5-4-3-2-1 Grounding Technique
        When experiencing acute anxiety or overwhelm, this exercise helps shift your focus away from your thoughts and back onto your physical senses.
        """)
        
        steps = [
            {"title": "👀 5 things you can SEE", "prompt": "Name five things in your field of vision (e.g., a chair, a plant, a spot on the wall...)"},
            {"title": "🤝 4 things you can TOUCH", "prompt": "Name four things you can feel physically (e.g., the texture of your shirt, the table surface, the floor under your feet...)"},
            {"title": "👂 3 things you can HEAR", "prompt": "Name three distinct sounds around you (e.g., traffic, ticking clock, wind, a humming fan...)"},
            {"title": "👃 2 things you can SMELL", "prompt": "Name two scents you can perceive or recall (e.g., coffee, soap, flowers, fresh air...)"},
            {"title": "👅 1 thing you can TASTE", "prompt": "Name one thing you can taste or focus on the current sensation in your mouth (e.g., mint, water, a recent meal...)"}
        ]
        
        curr_step = st.session_state.grounding_step
        
        if curr_step == 0:
            st.markdown('<div class="glass-card" style="text-align: center; padding: 40px 20px;">', unsafe_allow_html=True)
            st.markdown("### Ready to begin grounding?")
            st.markdown("This will guide you step-by-step to calm your nervous system.")
            if st.button("Start Grounding Exercise", use_container_width=True):
                st.session_state.grounding_step = 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif 1 <= curr_step <= 5:
            step_data = steps[curr_step - 1]
            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(step_data["title"])
            st.write(step_data["prompt"])
            
            ans_key = f"step_{curr_step}"
            user_ans = st.text_area("Write them down:", key=ans_key)
            
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                if st.button("Back", use_container_width=True):
                    st.session_state.grounding_step -= 1
                    st.rerun()
            with col_b2:
                if st.button("Next", use_container_width=True):
                    if user_ans.strip():
                        st.session_state.grounding_answers[ans_key] = user_ans
                        st.session_state.grounding_step += 1
                        st.rerun()
                    else:
                        st.warning("Please type something before proceeding.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif curr_step == 6:
            st.markdown('<div class="glass-card" style="text-align: center; padding: 30px;">', unsafe_allow_html=True)
            st.markdown("## 🎉 Grounding Complete!")
            st.markdown("Excellent work. Take one more deep, slow breath. Notice how you feel now compared to before.")
            
            # Show summarized grounding reflections
            st.markdown("### 📝 Your Reflections:")
            for i in range(1, 6):
                step_title = steps[i-1]["title"].split(" ")[1:]
                step_title = " ".join(step_title)
                st.markdown(f"**{i}. {step_title}**: *{st.session_state.grounding_answers.get(f'step_{i}', '')}*")
                
            if st.button("Finish & Reset", use_container_width=True):
                st.session_state.grounding_step = 0
                st.session_state.grounding_answers = {}
                st.balloons()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE 4: DAILY REFLECTION JOURNAL ---------------- #

elif page == "📖 Daily Reflection Journal":
    st.markdown('<div class="glass-header"><h1>📖 Daily Reflection Journal</h1><p>Pour your thoughts onto the screen. We will help you analyze the emotional tone. 💙</p></div>', unsafe_allow_html=True)
    
    col_j1, col_j2 = st.columns([2, 1])
    
    with col_j1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Write a new entry")
        entry_text = st.text_area("Write freely. What's on your mind? What went well? What was challenging?", height=250)
        
        if st.button("Analyze & Save Journal Entry", use_container_width=True):
            if entry_text.strip():
                # Detect sentiment/emotion
                detected_emo = get_detected_emotion(entry_text)
                
                # Save entry
                st.session_state.journal_entries.append({
                    "time": datetime.now(),
                    "entry": entry_text,
                    "emotion": detected_emo
                })
                
                # Log to mood tracker
                st.session_state.mood_history.append({
                    "time": datetime.now(),
                    "emotion": detected_emo,
                    "notes": f"Journal reflection: '{entry_text[:25]}...'"
                })
                
                st.success("Reflection saved!")
                st.balloons()
                st.rerun()
            else:
                st.warning("Please write something before saving.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_j2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💡 Reflection Prompt")
        
        prompts = [
            "What are three small things you are grateful for today?",
            "Describe a moment today when you felt peaceful or happy.",
            "If you could talk to your future self, what comforting words would you say?",
            "What is a boundary you successfully held today?",
            "What did you learn about yourself from a difficult situation recently?"
        ]
        
        # Select daily prompt based on the day of month
        prompt_idx = datetime.now().day % len(prompts)
        st.info(prompts[prompt_idx])
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Past entries list
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📜 Saved Reflections")
    if st.session_state.journal_entries:
        for idx, entry in enumerate(reversed(st.session_state.journal_entries)):
            time_str = entry["time"].strftime("%B %d, %Y at %I:%M %p")
            emoji = emotion_emojis.get(entry["emotion"], "😐")
            
            with st.expander(f"📖 Entry on {time_str} | Mood: {emoji} {entry['emotion'].capitalize()}"):
                st.write(entry["entry"])
                st.markdown(f"""
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 0.85rem; color: #9CA3AF;">
                    <strong>AI Mood Insights:</strong> This writing reflects feelings of <em>{entry['emotion']}</em>. Remember to take care of yourself.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Your journal entries will be listed here. Start typing above!")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE 5: SELF-CARE TRACKER ---------------- #

elif page == "🎯 Self-Care Tracker":
    st.markdown('<div class="glass-header"><h1>🎯 Daily Self-Care Tracker</h1><p>Small habits compound into major changes. Track your goals. 💙</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Goals Checklist")
    st.write("Complete these simple activities daily to boost emotional resilience.")
    
    # Render interactive checklist
    completed_count = 0
    updated_checklist = {}
    
    for habit, val in st.session_state.checklist.items():
        # Checkbox key needs to be unique and persistent
        checked = st.checkbox(habit, value=val)
        updated_checklist[habit] = checked
        if checked:
            completed_count += 1
            
    st.session_state.checklist = updated_checklist
    
    # Calculate progress
    total_habits = len(st.session_state.checklist)
    progress_percentage = int((completed_count / total_habits) * 100)
    
    st.markdown("### Progress")
    st.progress(completed_count / total_habits)
    st.write(f"**{progress_percentage}% Completed** ({completed_count} of {total_habits} goals)")
    
    if completed_count == total_habits:
        st.snow()
        st.markdown("""
        <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px dashed #10B981; padding: 15px; border-radius: 12px; text-align: center; margin-top: 15px;">
            🎉 <strong>Perfect Day!</strong> You completed all your self-care goals today. Celebrate your resilience!
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE 6: CALMING SOUNDSCAPES ---------------- #

elif page == "🎵 Calming Soundscapes":
    st.markdown('<div class="glass-header"><h1>🎵 Calming Soundscapes</h1><p>Play ambient sounds in the background to help you focus or relax. 💙</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    Select an ambient sound to play while you reflect, journal, or chat. These sounds play directly in your browser.
    """)
    
    col_s1, col_s2 = st.columns([1, 1])
    
    soundscapes = [
        {"name": "🌧️ Gentle Summer Rain", "url": "https://assets.mixkit.co/active_storage/sfx/2458/2458-84.wav"},
        {"name": "🌊 Ocean Waves", "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"}, # Relaxing instrumental as backup
        {"name": "🌲 Forest Nature Ambient", "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3"},
        {"name": "🎹 Soft Piano Lofi", "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3"}
    ]
    
    for idx, sound in enumerate(soundscapes):
        target_col = col_s1 if idx % 2 == 0 else col_s2
        with target_col:
            st.markdown(f'<div class="glass-card hover-card">', unsafe_allow_html=True)
            st.markdown(f"### {sound['name']}")
            st.audio(sound['url'], format="audio/wav")
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE 7: CRISIS CENTER ---------------- #

elif page == "🚨 Crisis & Resource Center":
    st.markdown('<div class="glass-header"><h1>🚨 Crisis & Help Center</h1><p>If you are in distress, professional human support is always available. 💙</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ZenFlow is an AI assistant, not a replacement for professional clinical care. If you are experiencing severe emotional distress, please connect with these free, confidential crisis resources:
    """)
    
    col_r1, col_r2 = st.columns([1, 1])
    
    with col_r1:
        st.markdown("""
        <div class="glass-card hover-card">
            <h3>🇺🇸 United States & Canada</h3>
            <p><strong>988 Suicide & Crisis Lifeline</strong></p>
            <p>Call or text <strong>988</strong> (Available 24/7, free, confidential)</p>
            <p><a href="https://988lifeline.org/" target="_blank">988lifeline.org</a></p>
            <hr style="border-top: 1px solid rgba(255,255,255,0.05)">
            <p><strong>The Crisis Text Line</strong></p>
            <p>Text <strong>HOME</strong> to <strong>741741</strong></p>
            <p><a href="https://www.crisistextline.org/" target="_blank">crisistextline.org</a></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r2:
        st.markdown("""
        <div class="glass-card hover-card">
            <h3>🇬🇧 United Kingdom</h3>
            <p><strong>Samaritans Helplines</strong></p>
            <p>Call <strong>116 123</strong> (Available 24/7, free)</p>
            <p><a href="https://www.samaritans.org/" target="_blank">samaritans.org</a></p>
            <hr style="border-top: 1px solid rgba(255,255,255,0.05)">
            <p><strong>Shout Crisis Text Line</strong></p>
            <p>Text <strong>SHOUT</strong> to <strong>85258</strong></p>
            <p><a href="https://giveusashout.org/" target="_blank">giveusashout.org</a></p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- GLOBAL DISCLAIMER FOOTER ---------------- #

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Disclaimer:</strong> ZenFlow is an AI-powered conversational support companion. It is designed to provide mindfulness exercises, general coping tips, and RAG-based context from self-help documents. It does not provide medical diagnostics, clinical therapy, or medical prescriptions. If you are experiencing a mental health emergency, please seek immediate help from a healthcare provider or a licensed counselor.
</div>
""", unsafe_allow_html=True)
