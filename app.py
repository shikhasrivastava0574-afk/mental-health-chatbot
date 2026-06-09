import os
from datetime import datetime
import pandas as pd
import gradio as gr

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

# ---------------- MODELS & DATA LOADING (CACHED) ---------------- #

# Check environment variable first
groq_api_key = os.getenv("GROQ_API_KEY")

# Load Hugging Face emotion classifier locally
try:
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
except Exception as e:
    print(f"Error loading emotion model: {e}")
    emotion_model = None

# Load FAISS Vector Store
pdf_folder = "data/pdfs"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
vectorstore = None

if not pdf_files:
    # Fallback default document
    from langchain.docstore.document import Document
    docs = [Document(
        page_content="ZenFlow is a supportive AI companion. For stress relief, focus on deep breathing, mindfulness, daily hydration, physical movement, and regular sleep.",
        metadata={"source": "system_default"}
    )]
else:
    documents = []
    for file in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF {file}: {e}")
            
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
except Exception as e:
    print(f"Error initializing vectorstore: {e}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

# Helper to initialize RAG QA Chain
def get_qa_chain(api_key):
    if not api_key or not retriever:
        return None
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",
            temperature=0.3
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )
    except Exception as e:
        print(f"Error creating LLM: {e}")
        return None

# ---------------- HYBRID EMOTION CLASSIFICATION ---------------- #

emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😟",
    "surprise": "😮",
    "love": "🥰",
    "neutral": "😐",
    "disgust": "🤢"
}

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

# ---------------- CUSTOM CSS FOR THEME & ANIMATIONS ---------------- #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

.gradio-container {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphism containers */
.glass-box {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
}

.header-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 25px;
    text-align: center;
    color: white;
}

.emotion-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    color: white;
    text-transform: capitalize;
    gap: 6px;
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
    width: 145px;
    height: 145px;
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

.history-item {
    padding: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.disclaimer-box {
    border-left: 4px solid #F59E0B;
    background-color: rgba(245, 158, 11, 0.08);
    padding: 15px;
    border-radius: 0 12px 12px 0;
    margin-top: 30px;
    color: #F59E0B;
}
"""

# ---------------- GRADIO LOGIC FUNCTIONS ---------------- #

def process_chat(user_message, chat_history, api_key_state, mood_history_state):
    if not user_message.strip():
        return "", chat_history, "", mood_history_state
        
    detected_emotion = get_detected_emotion(user_message)
    emoji = emotion_emojis.get(detected_emotion, "😐")
    
    # Save to mood history
    new_mood = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": detected_emotion,
        "notes": f"Detected in chat: '{user_message[:30]}...'"
    }
    updated_moods = mood_history_state + [new_mood]
    
    # Initialize RAG chain if possible
    active_key = api_key_state if api_key_state else groq_api_key
    qa_chain = get_qa_chain(active_key)
    
    if qa_chain:
        try:
            response = qa_chain.run(user_message)
        except Exception as e:
            response = f"I apologize, but I encountered an error: {e}."
    else:
        # Fallback responses based on emotion
        fallback_responses = {
            "joy": "I am so happy to hear that! It sounds like you are feeling joyful. Capturing and appreciating these moments of happiness is wonderful for your mental well-being. What is making you feel this way?",
            "sadness": "I hear you, and I am so sorry you are feeling sad. Please know that it is completely okay to feel this way, and you are not alone. Be gentle with yourself today. Would you like to try our Box Breathing or Grounding exercise in the menu?",
            "anger": "It sounds like you're carrying some anger or frustration, and that is completely valid. It can help to express these feelings. If you need a moment to release tension, perhaps we can do a brief Box Breathing session together.",
            "fear": "It feels like you are experiencing some anxiety or fear right now. Take a slow, deep breath. You are safe in this moment. I highly recommend checking out our Grounding Technique to help bring you back to the present.",
            "love": "What a beautiful feeling. Love and connection are so core to our resilience. Thank you for sharing that warm energy with me! Tell me more about it.",
            "surprise": "Oh, that sounds like a surprising turn of events! How are you processing this change? I'm here if you want to talk it through.",
            "neutral": "Thank you for sharing that with me. I'm here to listen. Tell me more about how you're feeling or what has been happening today.",
            "disgust": "I understand that you're feeling a sense of disgust or aversion. That can be a very uncomfortable emotion to sit with. Let's take a slow breath together and try to release some of that tension."
        }
        suggest_key = " (Note: Add your GROQ API Key in the sidebar or configuration to enable full conversational AI therapy powered by your PDFs!)"
        response = fallback_responses.get(detected_emotion, fallback_responses["neutral"]) + suggest_key

    # Append to chatbot history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})
    
    # Emotion badge markdown
    badge_html = f"""
    <div style="text-align: right; margin-top: 10px;">
        <span style="font-size: 0.9rem; color: #9CA3AF; margin-right: 8px;">Last Message Tone:</span>
        <span class="emotion-badge emotion-{detected_emotion}">{emoji} {detected_emotion}</span>
    </div>
    """
    
    return "", chat_history, badge_html, updated_moods

def apply_api_key(key):
    if key.strip():
        # Test initialization
        qa = get_qa_chain(key)
        if qa:
            return gr.update(visible=False), gr.update(value="✅ API Key applied and validated!"), key
        else:
            return gr.update(visible=True), gr.update(value="❌ Invalid or non-functional API Key. Please verify."), ""
    return gr.update(visible=True), gr.update(value=""), ""

def get_mood_analytics(mood_history):
    if not mood_history:
        return None, "<p>No mood logs yet.</p>"
        
    df = pd.DataFrame(mood_history)
    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["Emotion", "Count"]
    
    # Generate HTML history log
    log_html = ""
    for _, row in df.iloc[::-1].iterrows():
        emoji = emotion_emojis.get(row['emotion'], "😐")
        log_html += f"""
        <div class="history-item">
            <div>
                <span style="font-size: 1.15rem; margin-right: 8px;">{emoji}</span>
                <strong style="text-transform: capitalize;">{row['emotion']}</strong>
                <span style="color: #9CA3AF; margin-left: 15px; font-size: 0.9rem;">{row['notes']}</span>
            </div>
            <div style="color: #6B7280; font-size: 0.85rem;">{row['time']}</div>
        </div>
        """
        
    return counts, log_html

def manual_log_mood(emotion, notes, mood_history):
    new_mood = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion,
        "notes": notes if notes.strip() else "Manual Log"
    }
    updated_history = mood_history + [new_mood]
    counts_df, log_html = get_mood_analytics(updated_history)
    return counts_df, log_html, updated_history, ""

# Grounding wizard state updates
def process_grounding(step, text_val, answers_state):
    steps = [
        {"title": "👀 5 things you can SEE", "prompt": "Name five things in your field of vision:"},
        {"title": "🤝 4 things you can TOUCH", "prompt": "Name four things you can feel physically:"},
        {"title": "👂 3 things you can HEAR", "prompt": "Name three distinct sounds around you:"},
        {"title": "👃 2 things you can SMELL", "prompt": "Name two scents you can perceive or recall:"},
        {"title": "👅 1 thing you can TASTE", "prompt": "Name one thing you can taste:"}
    ]
    
    # Save answer of completed step
    if 1 <= step <= 5:
        answers_state[f"step_{step}"] = text_val
        
    next_step = step + 1
    
    # Generate visibility states for steps 0 to 6
    visibilities = [False] * 7
    visibilities[next_step] = True
    
    # If final step, compile reflections
    reflections = ""
    if next_step == 6:
        reflections = "### 📝 Your Grounding Reflections:\n"
        for i in range(1, 6):
            step_title = steps[i-1]["title"].split(" ")[1:]
            step_title = " ".join(step_title)
            reflections += f"**{i}. {step_title}**: *{answers_state.get(f'step_{i}', '')}*\n\n"
            
    return (
        gr.update(visible=visibilities[0]),
        gr.update(visible=visibilities[1]),
        gr.update(visible=visibilities[2]),
        gr.update(visible=visibilities[3]),
        gr.update(visible=visibilities[4]),
        gr.update(visible=visibilities[5]),
        gr.update(visible=visibilities[6]),
        next_step,
        "",  # clear input textbox
        reflections,
        answers_state
    )

def reset_grounding():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        0,
        "",
        "",
        {}
    )

# Reflection Journal Functions
def save_journal_entry(text, entries_state, mood_history):
    if not text.strip():
        return "⚠️ Please write something before saving.", entries_state, mood_history, ""
        
    detected_emo = get_detected_emotion(text)
    emoji = emotion_emojis.get(detected_emo, "😐")
    time_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    new_entry = {
        "time": time_str,
        "entry": text,
        "emotion": detected_emo
    }
    updated_entries = entries_state + [new_entry]
    
    # Log to mood tracker
    new_mood = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": detected_emo,
        "notes": f"Journal entry: '{text[:25]}...'"
    }
    updated_mood_history = mood_history + [new_mood]
    
    # Compile past entries HTML
    past_html = ""
    for entry in reversed(updated_entries):
        entry_emoji = emotion_emojis.get(entry["emotion"], "😐")
        past_html += f"""
        <div class="glass-box" style="margin-top: 15px;">
            <h4>📖 Entry on {entry['time']} | Mood: {entry_emoji} <span style="text-transform: capitalize;">{entry['emotion']}</span></h4>
            <p>{entry['entry']}</p>
            <hr style="margin: 10px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #9CA3AF;">
                <strong>AI Mood Insights:</strong> This writing reflects feelings of <em>{entry['emotion']}</em>. Remember to take care of yourself.
            </div>
        </div>
        """
        
    analysis_feedback = f"""
    ### Analysis Complete! 🎉
    **Detected Emotion:** {emoji} **{detected_emo.capitalize()}**
    
    *ZenFlow Insights:* Based on your reflection, we detected feelings of {detected_emo}. Keep reflecting and being gentle with yourself!
    """
    
    return analysis_feedback, updated_entries, updated_mood_history, past_html

def get_daily_reflection_prompt():
    prompts = [
        "What are three small things you are grateful for today?",
        "Describe a moment today when you felt peaceful or happy.",
        "If you could talk to your future self, what comforting words would you say?",
        "What is a boundary you successfully held today?",
        "What did you learn about yourself from a difficult situation recently?"
    ]
    prompt_idx = datetime.now().day % len(prompts)
    return f"**Today's reflection prompt:**\n*{prompts[prompt_idx]}*"

# Self-Care checklist functions
def update_checklist_progress(checked_list):
    total = 6
    completed = len(checked_list)
    pct = int((completed / total) * 100)
    
    progress_html = f"""
    <div style="margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 5px;">
            <span>Checklist Progress</span>
            <span>{pct}% ({completed}/{total} completed)</span>
        </div>
        <div style="background-color: rgba(255,255,255,0.08); height: 12px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #6366F1, #3B82F6); width: {pct}%; height: 100%; border-radius: 6px; box-shadow: 0 0 10px rgba(99, 102, 241, 0.4);"></div>
        </div>
    </div>
    """
    
    celebration_message = ""
    if pct == 100:
        celebration_message = """
        <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px dashed #10B981; padding: 15px; border-radius: 12px; text-align: center; margin-top: 15px; color: #10B981;">
            🎉 <strong>Perfect Day!</strong> You completed all your self-care goals today. Celebrate your resilience!
        </div>
        """
        
    return progress_html, celebration_message

# ---------------- GRAPHICAL INTERFACE (GRADIO BLOCKS) ---------------- #

with gr.Blocks() as demo:
    
    # States
    mood_history = gr.State([{"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "emotion": "neutral", "notes": "Welcome to ZenFlow!"}])
    journal_entries = gr.State([])
    grounding_step = gr.State(0)
    grounding_answers = gr.State({})
    api_key_state = gr.State(groq_api_key if groq_api_key else "")

    # Header Box
    gr.HTML("""
    <div class="header-box">
        <h1 style="font-size: 2.25rem; font-weight: 700; margin-bottom: 5px;">🌿 ZenFlow</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Your Interactive AI Mental Companion & Mindfulness Space 💙</p>
    </div>
    """)
    
    with gr.Row():
        
        # Sidebar/Left Column: API Settings & Quick Status
        with gr.Column(scale=1):
            with gr.Column(elem_classes="glass-box"):
                gr.Markdown("### 🔑 API Configuration")
                
                # Check status
                if groq_api_key:
                    gr.Markdown("✅ **GROQ API Key loaded from environment**")
                    api_status = gr.Markdown("")
                    api_input = gr.Textbox(visible=False)
                    api_btn = gr.Button("Apply Key", visible=False)
                else:
                    api_input = gr.Textbox(label="Enter GROQ API Key", type="password", placeholder="gsk_...")
                    api_status = gr.Markdown("⚠️ GROQ Key missing. RAG Chat will fallback to offline empathy responses.")
                    api_btn = gr.Button("Validate & Apply Key")
                
                # Navigation Help
                gr.Markdown("""
                ---
                ### 🧘 Self-Care Spaces
                Explore tabs above:
                - **🌿 AI Care Space:** Chat assistant with tone detection.
                - **📊 Mood Analytics:** Log & view emotion trends.
                - **🧘 Breathing & Grounding:** Relaxation guides.
                - **📖 Reflection Journal:** Diary & sentiment analytics.
                - **🎯 Self-Care Tracker:** Daily wellness checklist.
                - **🎵 Calming Soundscapes:** Ambient audio helper.
                """)
                
        # Main content / Right Column
        with gr.Column(scale=3):
            with gr.Tabs():
                
                # Tab 1: AI Care Space (Chat)
                with gr.TabItem("🌿 AI Care Space"):
                    gr.Markdown("### AI Care Chat assistant")
                    gr.Markdown("Speak freely about your feelings. ZenFlow classifies message sentiment and replies empathetically.")
                    
                    last_emotion_badge = gr.HTML("""
                    <div style="text-align: right; margin-top: 10px;">
                        <span style="font-size: 0.9rem; color: #9CA3AF; margin-right: 8px;">Last Message Tone:</span>
                        <span class="emotion-badge emotion-neutral">😐 neutral</span>
                    </div>
                    """)
                    
                    chatbot = gr.Chatbot(elem_id="chatbot", label="ZenFlow Assistant")
                    
                    with gr.Row():
                        chat_msg = gr.Textbox(placeholder="How are you feeling right now?", scale=4)
                        submit_btn = gr.Button("Submit", scale=1)
                        
                    gr.Markdown("*Note:ZenFlow search-matches contextual guides from database PDFs to provide optimal support.*")
                    
                # Tab 2: Mood Analytics
                with gr.TabItem("📊 Mood Analytics"):
                    gr.Markdown("### Log & Track Your Mood")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Log current state")
                            mood_input = gr.Dropdown(
                                choices=["neutral", "joy", "sadness", "anger", "fear", "surprise", "love"],
                                label="How do you feel?"
                            )
                            mood_notes = gr.Textbox(placeholder="Notes (e.g. 'had a nice walk')", label="Notes")
                            log_mood_btn = gr.Button("Save Mood Entry", variant="primary")
                            
                        with gr.Column(scale=2):
                            gr.Markdown("#### Emotion Distribution")
                            mood_chart = gr.BarPlot(
                                value=pd.DataFrame(columns=["Emotion", "Count"]),
                                x="Emotion",
                                y="Count",
                                title="Logged Emotions"
                            )
                            
                    gr.Markdown("#### 📜 Historical Logs")
                    mood_history_log = gr.HTML("<p>Log some moods to view your history.</p>")

                # Tab 3: Breathing & Grounding
                with gr.TabItem("🧘 Breathing & Grounding"):
                    gr.Markdown("### Calming Mindfulness Activities")
                    
                    with gr.Tabs():
                        # Sub-tab: Box Breathing
                        with gr.Tab("🌀 Box Breathing Visualizer"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("""
                                    ### What is Box Breathing?
                                    Box breathing is a powerful breathing exercise used to clear the mind, relax the body, and regulate stress.
                                    
                                    #### Rhythm Guide (4-4-4-4):
                                    1. **Inhale** slowly through your nose for 4 seconds.
                                    2. **Hold** your breath in for 4 seconds.
                                    3. **Exhale** completely through your mouth for 4 seconds.
                                    4. **Hold** your empty lungs for 4 seconds.
                                    5. Repeat this cycle 4 times or until calm.
                                    """)
                                    
                                with gr.Column(elem_classes="glass-box"):
                                    gr.Markdown("#### Sync your breathing with the glowing bubble:")
                                    gr.HTML("""
                                    <div class="breathing-wrapper">
                                        <div class="breathing-circle"></div>
                                    </div>
                                    """)
                                    
                        # Sub-tab: Grounding Game
                        with gr.Tab("🎯 5-4-3-2-1 Grounding Game"):
                            gr.Markdown("""
                            ### 🎯 5-4-3-2-1 Grounding Technique
                            When experiencing panic, stress, or high anxiety, use this exercise to draw your awareness away from thoughts and anchor yourself back to your body's physical senses.
                            """)
                            
                            # Steps Boxes
                            step_boxes = []
                            
                            # Step 0: Welcome
                            with gr.Group(visible=True) as box_0:
                                gr.Markdown("### Ready to begin grounding?")
                                gr.Markdown("This interactive guide will walk you step-by-step through checking your senses.")
                                start_grounding_btn = gr.Button("Start Exercise", variant="primary")
                            step_boxes.append(box_0)
                            
                            # Step 1: See
                            with gr.Group(visible=False) as box_1:
                                gr.Markdown("### 👀 5 things you can SEE")
                                gr.Markdown("Look around you. Name five distinct objects you can visually perceive:")
                                text_1 = gr.Textbox(placeholder="1. My laptop, 2. A green leaf, ...")
                                btn_next_1 = gr.Button("Next")
                            step_boxes.append(box_1)
                            
                            # Step 2: Touch
                            with gr.Group(visible=False) as box_2:
                                gr.Markdown("### 🤝 4 things you can TOUCH")
                                gr.Markdown("Feel your surroundings. Name four textures or temperatures you can physically touch:")
                                text_2 = gr.Textbox(placeholder="1. Warm tea mug, 2. Denim fabric, ...")
                                btn_next_2 = gr.Button("Next")
                            step_boxes.append(box_2)
                            
                            # Step 3: Hear
                            with gr.Group(visible=False) as box_3:
                                gr.Markdown("### 👂 3 things you can HEAR")
                                gr.Markdown("Listen carefully. Name three distinct sounds in your environment:")
                                text_3 = gr.Textbox(placeholder="1. Fan humming, 2. Distance bird chirping, ...")
                                btn_next_3 = gr.Button("Next")
                            step_boxes.append(box_3)
                            
                            # Step 4: Smell
                            with gr.Group(visible=False) as box_4:
                                gr.Markdown("### 👃 2 things you can SMELL")
                                gr.Markdown("Breathe in. Name two scents you can perceive or recall:")
                                text_4 = gr.Textbox(placeholder="1. Coffee aroma, 2. Rain, ...")
                                btn_next_4 = gr.Button("Next")
                            step_boxes.append(box_4)
                            
                            # Step 5: Taste
                            with gr.Group(visible=False) as box_5:
                                gr.Markdown("### 👅 1 thing you can TASTE")
                                gr.Markdown("Name one thing you can taste, or focus on the current physical sensation inside your mouth:")
                                text_5 = gr.Textbox(placeholder="1. Mint toothpaste flavor, ...")
                                btn_next_5 = gr.Button("Complete Grounding")
                            step_boxes.append(box_5)
                            
                            # Step 6: Complete
                            with gr.Group(visible=False) as box_6:
                                gr.Markdown("## 🎉 Grounding Completed!")
                                gr.Markdown("Take a slow, deep breath. Great job centering your awareness.")
                                grounding_summary = gr.Markdown("")
                                reset_grounding_btn = gr.Button("Finish & Reset")
                            step_boxes.append(box_6)

                # Tab 4: Reflection Journal
                with gr.TabItem("📖 Daily Reflection Journal"):
                    gr.Markdown("### Daily Reflection Journal")
                    gr.Markdown("Write about your day. ZenFlow analyzes the emotional tone and saves it to your journal logs.")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            journal_input = gr.Textbox(
                                placeholder="Write freely about your reflections...", 
                                label="Journal entry", 
                                lines=8
                            )
                            save_journal_btn = gr.Button("Analyze & Save Reflection", variant="primary")
                        with gr.Column(scale=1):
                            journal_prompt = gr.Markdown(get_daily_reflection_prompt())
                            journal_analysis_box = gr.Markdown("Write an entry to view sentiment analysis results.")
                            
                    gr.Markdown("### 📜 Past Reflections")
                    past_journal_list = gr.HTML("<p>Your journal entries will be listed here.</p>")

                # Tab 5: Daily Self-Care Tracker
                with gr.TabItem("🎯 Self-Care Tracker"):
                    gr.Markdown("### Daily Self-Care Goals")
                    gr.Markdown("Complete daily check-ins to build psychological resilience.")
                    
                    checklist_input = gr.CheckboxGroup(
                        choices=[
                            "💧 Drink 8 glasses of water",
                            "🚶 Take a 15-minute walk",
                            "🧘 Do a breathing exercise",
                            "📖 Write in my journal",
                            "🛌 Sleep 8 hours",
                            "🍎 Eat a healthy meal"
                        ],
                        label="Wellness checklist"
                    )
                    
                    checklist_progress = gr.HTML("""
                    <div style="margin: 15px 0;">
                        <div style="display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 5px;">
                            <span>Checklist Progress</span>
                            <span>0% (0/6 completed)</span>
                        </div>
                        <div style="background-color: rgba(255,255,255,0.08); height: 12px; border-radius: 6px; overflow: hidden;">
                            <div style="background-color: #6366F1; width: 0%; height: 100%; border-radius: 6px;"></div>
                        </div>
                    </div>
                    """)
                    
                    checklist_celebrate = gr.HTML("")

                # Tab 6: Calming Soundscapes
                with gr.TabItem("🎵 Calming Soundscapes"):
                    gr.Markdown("### Calming Soundscapes")
                    gr.Markdown("Play background ambient sounds while you reflect, journal, or write.")
                    
                    with gr.Row():
                        with gr.Column(elem_classes="glass-box"):
                            gr.HTML("<h4>🌧️ Gentle Summer Rain</h4>")
                            gr.HTML('<audio src="https://assets.mixkit.co/active_storage/sfx/2458/2458-84.wav" controls style="width: 100%;"></audio>')
                        with gr.Column(elem_classes="glass-box"):
                            gr.HTML("<h4>🌲 Forest Nature Ambient</h4>")
                            gr.HTML('<audio src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3" controls style="width: 100%;"></audio>')
                    with gr.Row():
                        with gr.Column(elem_classes="glass-box"):
                            gr.HTML("<h4>🌊 Ocean Waves (Instrumental)</h4>")
                            gr.HTML('<audio src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3" controls style="width: 100%;"></audio>')
                        with gr.Column(elem_classes="glass-box"):
                            gr.HTML("<h4>🎹 Soft Piano Lofi</h4>")
                            gr.HTML('<audio src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3" controls style="width: 100%;"></audio>')

                # Tab 7: Crisis Resource Center
                with gr.TabItem("🚨 Crisis & Resource Center"):
                    gr.Markdown("### 🚨 Emergency Resources")
                    gr.Markdown("ZenFlow is an AI companion and does not substitute professional medical care. If you are experiencing distress, please reach out to one of the following:")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            ### 🇺🇸 United States & Canada
                            - **988 Suicide & Crisis Lifeline**: Call or text **988** (Available 24/7, free, confidential) | [988lifeline.org](https://988lifeline.org/)
                            - **Crisis Text Line**: Text **HOME** to **741741** | [crisistextline.org](https://www.crisistextline.org/)
                            """)
                        with gr.Column():
                            gr.Markdown("""
                            ### 🇬🇧 United Kingdom
                            - **Samaritans**: Call **116 123** (Available 24/7, free) | [samaritans.org](https://www.samaritans.org/)
                            - **Shout**: Text **SHOUT** to **85258** | [giveusashout.org](https://giveusashout.org/)
                            """)

    # Medical Disclaimer Footer
    gr.HTML("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer:</strong> ZenFlow is an AI-powered mindfulness support companion. It is designed to guide you through meditation exercises, coping tips, and search self-help books. It does not provide medical therapy, prescriptions, or emergency suicide intervention. If you are experiencing distress, please reach out to a licensed professional counselor.
    </div>
    """)

    # ---------------- INTERACTIVITY BINDINGS ---------------- #

    # API Validation button
    if not groq_api_key:
        api_btn.click(
            fn=apply_api_key,
            inputs=[api_input],
            outputs=[api_btn, api_status, api_key_state]
        )

    # Chat submit action
    chat_inputs = [chat_msg, chatbot, api_key_state, mood_history]
    chat_outputs = [chat_msg, chatbot, last_emotion_badge, mood_history]
    
    submit_btn.click(fn=process_chat, inputs=chat_inputs, outputs=chat_outputs)
    chat_msg.submit(fn=process_chat, inputs=chat_inputs, outputs=chat_outputs)

    # Mood log save action
    log_mood_btn.click(
        fn=manual_log_mood,
        inputs=[mood_input, mood_notes, mood_history],
        outputs=[mood_chart, mood_history_log, mood_history, mood_notes]
    )

    # Grounding Wizard step transitions
    start_grounding_btn.click(
        fn=process_grounding,
        inputs=[grounding_step, gr.State(""), grounding_answers],
        outputs=step_boxes + [grounding_step, gr.State(""), grounding_summary, grounding_answers]
    )
    btn_next_1.click(
        fn=process_grounding,
        inputs=[grounding_step, text_1, grounding_answers],
        outputs=step_boxes + [grounding_step, text_1, grounding_summary, grounding_answers]
    )
    btn_next_2.click(
        fn=process_grounding,
        inputs=[grounding_step, text_2, grounding_answers],
        outputs=step_boxes + [grounding_step, text_2, grounding_summary, grounding_answers]
    )
    btn_next_3.click(
        fn=process_grounding,
        inputs=[grounding_step, text_3, grounding_answers],
        outputs=step_boxes + [grounding_step, text_3, grounding_summary, grounding_answers]
    )
    btn_next_4.click(
        fn=process_grounding,
        inputs=[grounding_step, text_4, grounding_answers],
        outputs=step_boxes + [grounding_step, text_4, grounding_summary, grounding_answers]
    )
    btn_next_5.click(
        fn=process_grounding,
        inputs=[grounding_step, text_5, grounding_answers],
        outputs=step_boxes + [grounding_step, text_5, grounding_summary, grounding_answers]
    )
    reset_grounding_btn.click(
        fn=reset_grounding,
        inputs=[],
        outputs=step_boxes + [grounding_step, text_1, grounding_summary, grounding_answers]
    )

    # Save Reflection Journal
    save_journal_btn.click(
        fn=save_journal_entry,
        inputs=[journal_input, journal_entries, mood_history],
        outputs=[journal_analysis_box, journal_entries, mood_history, past_journal_list]
    )

    # Checklist changes
    checklist_input.change(
        fn=update_checklist_progress,
        inputs=[checklist_input],
        outputs=[checklist_progress, checklist_celebrate]
    )
    
    # Initialize history list HTML on load
    demo.load(
        fn=get_mood_analytics,
        inputs=[mood_history],
        outputs=[mood_chart, mood_history_log]
    )

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="blue"), 
        css=CSS
    )
