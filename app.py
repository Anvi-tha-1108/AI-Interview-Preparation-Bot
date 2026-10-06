# ─────────────────────────────────────────────────────────────────────────────
# app.py  –  AI Interview Preparation Bot  |  Main Streamlit Application
#
# HOW STREAMLIT WORKS:
#   Streamlit re-runs this ENTIRE script from top to bottom every time:
#     • The user interacts with a widget (button, slider, text input)
#     • Something triggers a rerun (st.rerun())
#
#   This is why we use st.session_state to preserve data across reruns.
#   Think of each rerun as a "snapshot" of the UI.
#
# RUN:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

# ── Standard Library ──────────────────────────────────────────────────────────
import time
import os

# ── Third Party: Streamlit UI ─────────────────────────────────────────────────
import streamlit as st           # Web UI framework
import pandas as pd              # For displaying score history tables
import numpy as np               # Numerical operations

# ── Our Modules ───────────────────────────────────────────────────────────────
import config
from modules.question_generator import QuestionGenerator
from modules.answer_evaluator   import AnswerEvaluator
from modules.ai_coach           import AICoach
from modules.session_manager    import SessionManager


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIGURATION  (must be the FIRST Streamlit call)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = config.PAGE_TITLE,
    page_icon  = config.PAGE_ICON,
    layout     = config.LAYOUT,
    initial_sidebar_state = "expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  CUSTOM CSS  (injected once per load)
# ═════════════════════════════════════════════════════════════════════════════
# st.markdown() with unsafe_allow_html=True lets us inject raw HTML/CSS.
# We use this sparingly – only for styling that Streamlit doesn't expose natively.
st.markdown("""
<style>
/* Score badge styling */
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1.1em;
}
.score-excellent { background: #d4edda; color: #155724; }
.score-good      { background: #cce5ff; color: #004085; }
.score-average   { background: #fff3cd; color: #856404; }
.score-poor      { background: #f8d7da; color: #721c24; }

/* Question card */
.question-card {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  RESOURCE INITIALISATION  (cached so they run only once per session)
# ═════════════════════════════════════════════════════════════════════════════
# @st.cache_resource caches the OBJECT (not just its output).
# The function runs once; subsequent calls return the cached object.
# Perfect for heavy resources like ML models or database connections.

@st.cache_resource
def get_question_generator() -> QuestionGenerator:
    """Cached QuestionGenerator instance."""
    return QuestionGenerator()

@st.cache_resource
def get_answer_evaluator() -> AnswerEvaluator:
    """Cached AnswerEvaluator instance (loads spaCy model lazily)."""
    return AnswerEvaluator()

@st.cache_resource
def get_ai_coach() -> AICoach:
    """Cached AICoach instance."""
    return AICoach()


# ── Instantiate singletons ────────────────────────────────────────────────────
question_gen = get_question_generator()
evaluator    = get_answer_evaluator()
coach        = get_ai_coach()
session_mgr  = SessionManager()     # SessionManager wraps st.session_state directly


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SIDEBAR  –  Configuration panel
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # st.image() displays an image. Here we use an emoji fallback.
    st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
    st.divider()


    # ── Interview settings ─────────────────────────────────────────────────────
    st.subheader("⚙️ Session Settings")

    domain = st.selectbox(
        "Interview Domain",
        options = config.INTERVIEW_DOMAINS,
        index   = 0,
        help    = "Choose the role type to practice for",
    )

    difficulty = st.selectbox(
        "Difficulty Level",
        options = config.DIFFICULTY_LEVELS,
        index   = 1,     # Default: "Intermediate"
    )

    num_questions = st.slider(
        "Number of Questions",
        min_value = 3,
        max_value = 15,
        value     = 5,
        step      = 1,
    )

    topic_focus = st.text_input(
        "Topic Focus (optional)",
        placeholder = "e.g., linked lists, system design, pandas...",
        help        = "Leave blank for a general session",
    )

    st.divider()

    # ── TTS settings ───────────────────────────────────────────────────────────
    st.subheader("🔊 Voice Settings")
    enable_tts = st.toggle("Enable Text-to-Speech", value=False)
    use_gtts   = st.toggle("Use Google TTS (online)", value=False,
                            disabled=not enable_tts)

    st.divider()

    # ── Start / Reset buttons ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if reset_btn:
        session_mgr.reset()
        st.rerun()  # Trigger a fresh UI render

    # ── Session stats (mini dashboard in sidebar) ──────────────────────────────
    if session_mgr.entries:
        st.divider()
        st.subheader("📊 Session Stats")
        stats = session_mgr.get_summary_stats()
        st.metric("Avg Score",   f"{stats.get('mean_score', 0):.1f}/100")
        st.metric("Best Score",  f"{stats.get('max_score', 0):.1f}/100")
        st.metric("Questions",   stats.get("total_questions", 0))
        trend = stats.get("trend_label", "")
        if trend:
            st.info(trend)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  START SESSION (when the Start button is pressed)
# ═════════════════════════════════════════════════════════════════════════════
if start_btn:
    with st.spinner("🤖 Generating your personalised interview questions..."):
        questions = question_gen.generate(
            domain     = domain,
            difficulty = difficulty,
            count      = num_questions,
            topic      = topic_focus,
        )
    session_mgr.start_session(questions, domain, difficulty)
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN TABS
# ═════════════════════════════════════════════════════════════════════════════
# st.tabs() creates a tab bar. Each `with tab:` block renders inside that tab.
tab_interview, tab_chat, tab_progress, tab_guide = st.tabs([
    "🎯 Interview Practice",
    "💬 AI Chat Coach",
    "📈 My Progress",
    "📚 How to Use",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: INTERVIEW PRACTICE
# ─────────────────────────────────────────────────────────────────────────────
with tab_interview:

    if not session_mgr.is_started:
        # ── Welcome screen ─────────────────────────────────────────────────────
        st.markdown("""
        ## Welcome to your AI Interview Coach! 🤖

        **How it works:**
        1. 👈 Choose your **domain**, **difficulty**, and **question count** in the sidebar
        2. Click **▶ Start** to generate personalised questions
        3. Answer each question by **typing** or **recording your voice**
        4. Get instant **AI feedback** and a **score** for every answer
        5. Review your **progress** in the Progress tab

        ### What makes this bot special?
        - 🧠 **AI-generated questions** powered by Google Gemini
        - 🎙️ **Voice recognition** so you can practice speaking
        - 📊 **NLP scoring** using TF-IDF, NLTK, and spaCy
        - 💡 **Personalised coaching** with specific improvement tips
        """)

        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎯 **10 domains** from SWE to PM")
        with col2:
            st.info("🎙️ **Voice + text** input modes")
        with col3:
            st.info("📈 **Trend analysis** of your improvement")

    else:
        # ── Active interview session ────────────────────────────────────────────
        current_q = session_mgr.current_question
        total_q   = len(session_mgr.questions)
        done_q    = session_mgr.current_question_idx

        # ── Progress bar ───────────────────────────────────────────────────────
        progress = done_q / total_q if total_q > 0 else 0
        st.progress(progress)
        st.caption(f"Question {min(done_q + 1, total_q)} of {total_q}")

        # ── Session complete ───────────────────────────────────────────────────
        if current_q is None:
            st.balloons()   # Confetti animation!
            st.success("🎉 **Interview session complete!**")

            stats = session_mgr.get_summary_stats()
            st.metric("Final Average Score", f"{stats.get('mean_score', 0):.1f} / 100")

            # Download session report
            json_data = session_mgr.export_json()
            st.download_button(
                label     = "⬇️ Download Session Report (JSON)",
                data      = json_data,
                file_name = f"interview_report_{session_mgr.session_id[:8]}.json",
                mime      = "application/json",
            )

            if st.button("🔄 Start New Session"):
                session_mgr.reset()
                st.rerun()

        else:
            # ── Question display ───────────────────────────────────────────────
            st.markdown(f"""
<div class="question-card">
<strong>Q{done_q + 1}. [{current_q.difficulty}] [{current_q.category}]</strong><br><br>
{current_q.text}
</div>
""", unsafe_allow_html=True)

            # TTS: speak the question aloud
            if enable_tts and st.button("🔊 Hear Question"):
                from modules.speech_handler import SpeechHandler
                sh = SpeechHandler()
                audio_bytes = sh.speak_text(current_q.text, use_gtts=use_gtts)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3" if use_gtts else "audio/wav")

            # ── Hints expander ─────────────────────────────────────────────────
            if current_q.hints:
                with st.expander("💡 Show Hints"):
                    for hint in current_q.hints:
                        st.markdown(f"• {hint}")

            # ── Answer input ───────────────────────────────────────────────────
            input_mode = st.radio(
                "Answer mode",
                options=["✍️ Type", "🎙️ Voice (record .wav file)"],
                horizontal=True,
                label_visibility="collapsed",
            )

            answer_text = ""

            if "✍️" in input_mode:
                # Text area for typed answers
                answer_text = st.text_area(
                    "Your Answer",
                    height      = 200,
                    placeholder = "Type your answer here...\n\nTip: Aim for 50–150 words. Use the STAR method for behavioral questions.",
                )

            else:
                # Voice input: upload a recorded .wav file
                st.info("🎙️ Record your answer externally and upload the .wav file below.")
                audio_file = st.file_uploader(
                    "Upload audio answer (.wav or .mp3)",
                    type=["wav", "mp3"],
                )
                if audio_file is not None:
                    # Save uploaded file to temp dir
                    import tempfile, shutil
                    suffix = ".wav" if audio_file.name.endswith(".wav") else ".mp3"
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        dir=config.TEMP_AUDIO_DIR,
                        delete=False
                    ) as tmp:
                        shutil.copyfileobj(audio_file, tmp)
                        tmp_path = tmp.name

                    # Transcribe
                    from modules.speech_handler import SpeechHandler
                    sh = SpeechHandler()
                    with st.spinner("🎧 Transcribing..."):
                        answer_text, confidence = sh.transcribe_audio(tmp_path)

                    if answer_text:
                        st.success(f"✅ Transcribed (confidence: {confidence:.0%})")
                        st.markdown(f"**Your answer:** {answer_text}")
                    else:
                        st.warning("⚠️ Could not transcribe. Please type your answer.")

            # ── Submit button ──────────────────────────────────────────────────
            submit_col, skip_col = st.columns([3, 1])
            with submit_col:
                submit_btn = st.button(
                    "✅ Submit Answer",
                    type="primary",
                    use_container_width=True,
                    disabled=(not answer_text.strip()),
                )
            with skip_col:
                skip_btn = st.button("⏭️ Skip", use_container_width=True)

            if skip_btn:
                # Record a blank entry and advance
                from modules.answer_evaluator import EvaluationResult
                session_mgr.record_answer(
                    current_q,
                    "[Skipped]",
                    EvaluationResult(0, 0, 0, 0, 0, feedback="Question skipped."),
                )
                st.rerun()

            # ── Evaluation & Feedback ──────────────────────────────────────────
            if submit_btn and answer_text.strip():
                with st.spinner("🔍 Evaluating your answer..."):
                    evaluation = evaluator.evaluate(
                        answer       = answer_text,
                        ideal_answer = current_q.ideal_answer,
                        question_text = current_q.text,
                    )

                # ── Score display ──────────────────────────────────────────────
                st.divider()
                st.subheader("📊 Your Score")

                # 4-column metric row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall",      f"{evaluation.overall_score:.0f}/100")
                c2.metric("Relevance",    f"{evaluation.similarity_score:.0f}/100")
                c3.metric("Completeness", f"{evaluation.completeness_score:.0f}/100")
                c4.metric("Fluency",      f"{evaluation.fluency_score:.0f}/100")

                # Grade badge
                grade_class = {
                    "Excellent 🏆": "score-excellent",
                    "Good 👍":       "score-good",
                    "Average 📘":    "score-average",
                }.get(evaluation.grade, "score-poor")
                st.markdown(
                    f'<span class="score-badge {grade_class}">{evaluation.grade}</span>',
                    unsafe_allow_html=True,
                )

                # ── Detailed feedback ──────────────────────────────────────────
                with st.expander("📝 Detailed Feedback", expanded=True):
                    st.write(evaluation.feedback)

                    if evaluation.keywords_used:
                        st.success(f"✅ Key concepts mentioned: {', '.join(evaluation.keywords_used[:8])}")
                    if evaluation.keywords_missing:
                        st.warning(f"❌ Missed concepts: {', '.join(evaluation.keywords_missing[:8])}")
                    if evaluation.filler_words_found:
                        st.info(f"🗣️ Filler words detected: {', '.join(evaluation.filler_words_found)}")

                    # Improvement suggestions
                    if evaluation.suggestions:
                        st.subheader("💡 Suggestions")
                        for s in evaluation.suggestions:
                            st.markdown(f"• {s}")

                # ── Ideal answer reveal ────────────────────────────────────────
                with st.expander("📖 Model Answer"):
                    st.write(current_q.ideal_answer)

                # ── AI Coaching ────────────────────────────────────────────────
                st.subheader("🤖 AI Coach Feedback")
                coaching_placeholder = st.empty()
                # st.empty() creates a placeholder we can fill later.

                coaching_text = ""
                try:
                    with coaching_placeholder.container():
                        # st.write_stream() consumes a generator, displaying
                        # each yielded chunk as it arrives (typewriter effect).
                        coaching_text = st.write_stream(
                            coach.stream_coaching(current_q, answer_text, evaluation)
                        )
                except Exception as e:
                    coaching_text = coach.get_coaching(current_q, answer_text, evaluation)
                    coaching_placeholder.write(coaching_text)

                # Follow-up question
                if current_q.follow_up:
                    st.info(f"**Follow-up:** {current_q.follow_up}")
                    with st.expander("🧭 How to answer the follow-up"):
                        fu_coaching = coach.get_follow_up_coaching(
                            current_q, answer_text
                        )
                        st.write(fu_coaching)

                # ── Save and advance ───────────────────────────────────────────
                session_mgr.record_answer(
                    question      = current_q,
                    answer        = answer_text,
                    evaluation    = evaluation,
                    coaching_notes = coaching_text,
                )

                time.sleep(0.5)  # Brief pause for UX (lets the user read)

                # "Next Question" button
                if st.button("➡️ Next Question", type="primary"):
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: AI CHAT COACH
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("💬 Chat with your AI Interview Coach")
    st.caption("Ask anything: 'How do I answer STAR questions?', 'Explain Big-O', 'Mock interview me for Google'")

    # Display chat history
    # st.session_state["chat_history"] is a list of {"role", "content"} dicts
    for message in st.session_state.get("chat_history", []):
        # st.chat_message() renders a styled chat bubble
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input box (always at the bottom)
    if user_input := st.chat_input("Ask your coach anything..."):
        # := is the walrus operator (Python 3.8+).
        # It assigns AND tests in one expression:
        # "if there's a non-empty input, assign it to user_input AND enter the if block"

        # Show user's message immediately
        with st.chat_message("user"):
            st.write(user_input)
        session_mgr.add_chat_message("user", user_input)

        # Build Gemini conversation history format
        gemini_history = []
        for msg in st.session_state.get("chat_history", [])[:-1]:  # Exclude last msg
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]],
            })

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = coach.chat(gemini_history, user_input)
            st.write(response)

        session_mgr.add_chat_message("assistant", response)

    if not st.session_state.get("chat_history"):
        # Starter prompts to help users get going
        st.subheader("💡 Try asking:")
        starter_prompts = [
            "How do I use the STAR method for behavioral questions?",
            "Can you do a quick mock interview for a Python developer role?",
            "What's the best way to answer 'Tell me about yourself'?",
            "Explain how to approach a system design interview",
        ]
        for prompt in starter_prompts:
            if st.button(prompt, key=f"starter_{prompt[:20]}"):
                session_mgr.add_chat_message("user", prompt)
                with st.spinner("🤔 Thinking..."):
                    resp = coach.chat([], prompt)
                session_mgr.add_chat_message("assistant", resp)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PROGRESS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_progress:
    st.header("📈 Your Progress Dashboard")

    if not session_mgr.entries:
        st.info("Complete some interview questions to see your progress here.")
    else:
        df = session_mgr.get_analytics_df()
        stats = session_mgr.get_summary_stats()

        # ── KPI cards ──────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Average Score",    f"{stats.get('mean_score', 0):.1f}")
        k2.metric("Best Score",       f"{stats.get('max_score', 0):.1f}")
        k3.metric("Total Questions",  stats.get("total_questions", 0))
        k4.metric("Trend",            stats.get("trend_label", "N/A"))

        st.divider()

        # ── Score over time chart ───────────────────────────────────────────────
        st.subheader("Score Trend")
        # st.line_chart() is a quick chart – pass a Series or DataFrame column
        chart_df = df[["overall_score"]].rename(columns={"overall_score": "Score"})
        chart_df.index = [f"Q{i+1}" for i in range(len(chart_df))]
        st.line_chart(chart_df)

        # ── Sub-score breakdown ─────────────────────────────────────────────────
        st.subheader("Score Breakdown per Question")
        breakdown = df[["similarity_score", "fluency_score", "overall_score"]].copy()
        breakdown.columns = ["Relevance", "Fluency", "Overall"]
        breakdown.index = [f"Q{i+1}" for i in range(len(breakdown))]
        st.bar_chart(breakdown)

        # ── Performance by domain ───────────────────────────────────────────────
        by_domain = stats.get("by_domain", {})
        if by_domain:
            st.subheader("Average Score by Domain")
            domain_df = pd.DataFrame(
                list(by_domain.items()),
                columns=["Domain", "Avg Score"]
            ).set_index("Domain")
            st.bar_chart(domain_df)

        # ── Full history table ──────────────────────────────────────────────────
        st.subheader("Full Session History")
        display_df = df[["question_text", "domain", "difficulty",
                          "overall_score", "fluency_score"]].copy()
        display_df.columns = ["Question", "Domain", "Difficulty", "Score", "Fluency"]
        display_df["Question"] = display_df["Question"].str[:60] + "..."
        st.dataframe(display_df, use_container_width=True)

        # ── Download ────────────────────────────────────────────────────────────
        st.download_button(
            "⬇️ Export Full Report (JSON)",
            data      = session_mgr.export_json(),
            file_name = "interview_report.json",
            mime      = "application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: HOW TO USE GUIDE
# ─────────────────────────────────────────────────────────────────────────────
with tab_guide:
    st.header("📚 How to Use This Bot")
    st.markdown("""
    ## Quick Start Guide

    ### Step 1 – Configure
    Open the **sidebar** (👈 left panel) and:
    - Choose your **domain** and **difficulty**
    - Set how many questions you want

    ### Step 2 – Start
    Click **▶ Start** to generate AI-powered questions tailored to your settings.

    ### Step 3 – Answer
    Each question shows:
    - 📝 The question with **difficulty and category tags**
    - 💡 **Hints** (click to reveal)
    - ✍️ A **text area** to type your answer
    - 🎙️ A **voice upload** option (record externally, upload .wav)

    ### Step 4 – Get Feedback
    After submitting, you'll see:
    | Metric | What it measures |
    |---|---|
    | **Relevance** | How closely your answer matches the model answer (TF-IDF) |
    | **Completeness** | How many key concepts you covered (NLTK keyword analysis) |
    | **Fluency** | Penalty for filler words (um, like, basically…) |
    | **Confidence** | Length and assertiveness of your answer |
    | **Overall** | Weighted average of all four |

    ### Step 5 – Review Progress
    Visit the **📈 My Progress** tab to see:
    - Score trends over time
    - Domain-by-domain breakdown
    - Full session history exportable as JSON

    ---
    ## Technology Stack
    | Library | Role |
    |---|---|
    | `streamlit` | Web UI |
    | `google-genai` | Question generation + coaching (Gemini) |
    | `langchain` | Prompt templates for coaching |
    | `scikit-learn` | TF-IDF vectorisation + cosine similarity |
    | `nltk` | Tokenisation, stopwords, keyword extraction |
    | `spacy` | Named entity recognition |
    | `transformers` | Sentiment analysis (DistilBERT) |
    | `torch` | PyTorch backend for transformers |
    | `librosa` | Audio feature extraction |
    | `SpeechRecognition` | Speech-to-text |
    | `pyttsx3` / `gTTS` | Text-to-speech |
    | `pandas` | Session data management + analytics |
    | `numpy` | Numerical scoring computations |
    | `python-dotenv` | Secure API key management |

    ---
    ## Tips for Best Results
    - 🎯 Use the **STAR method** for behavioral questions: **S**ituation, **T**ask, **A**ction, **R**esult
    - ⏱️ Aim for **50–150 words** per answer (1–2 minutes spoken)
    - 📖 Always check the **Model Answer** – it's your reference
    - 🔁 **Repeat difficult questions** – start a new session with the same domain
    - 🗣️ Practice **speaking aloud** even when typing answers
    """)
