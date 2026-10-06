# 🤖 AI Interview Preparation Bot

> **Practice job interviews with real-time AI feedback, voice input, NLP scoring, and personalised coaching — powered by Google Gemini.**

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange?logo=google)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 AI Question Generation | Google Gemini generates fresh questions for your domain & difficulty |
| 🎙️ Voice Input | Upload a recorded .wav file; the bot transcribes it automatically |
| 📊 NLP Scoring | TF-IDF similarity + NLTK keyword analysis + spaCy NER |
| 💡 Personalised Coaching | Streaming AI feedback on every answer |
| 📈 Progress Tracking | Score trends over time using sklearn LinearRegression |
| 💬 Chat Coach | Free-form AI conversation for mock interviews & tips |
| 🔊 Text-to-Speech | Questions read aloud via gTTS or pyttsx3 |
| 📥 Export Reports | Download full session as JSON |

---

## 🗂️ Project Structure

```
AI Interview Preparation Bot Project/
│
├── app.py                   # Main Streamlit application (entry point)
├── config.py                # All settings & constants
├── setup.py                 # One-time setup script (downloads NLP models)
├── requirements.txt         # Python dependencies
├── .env                     # Your secret API keys (NOT committed to Git)
├── .env.example             # Template for .env (safe to commit)
├── .gitignore               # Files excluded from Git
│
├── modules/
│   ├── __init__.py          # Makes `modules` a Python package
│   ├── question_generator.py  # Gemini-powered question generation
│   ├── answer_evaluator.py    # NLP-based answer scoring
│   ├── ai_coach.py            # Streaming coaching + sentiment analysis
│   └── session_manager.py     # Session state + analytics
│
├── assets/                  # Static files (images, CSS)
└── temp_audio/              # Temporary audio files (auto-created, gitignored)
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-interview-prep-bot.git
cd ai-interview-prep-bot
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **PyAudio on Windows** may need a pre-built wheel:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

### 4. Set up your API key
```bash
# Copy the template
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux

# Edit .env and paste your Gemini key
# Get a FREE key at: https://aistudio.google.com/
```

### 5. Run one-time setup (downloads NLP models)
```bash
python setup.py
```

### 6. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 🔑 Getting Your Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy and paste it into `.env` or the sidebar input in the app

> The free tier is very generous — more than enough for practice sessions.

---

## 📦 Key Dependencies Explained

| Library | Version | Why We Use It |
|---|---|---|
| `streamlit` | ≥1.40 | Turns Python scripts into interactive web apps |
| `google-genai` | ≥1.0 | Official Gemini SDK for question gen + coaching |
| `langchain` | ≥0.3 | Prompt templates for structured coaching prompts |
| `scikit-learn` | ≥1.5 | TF-IDF vectorisation & cosine similarity scoring |
| `nltk` | ≥3.9 | Tokenisation, stopwords, keyword analysis |
| `spacy` | ≥3.7 | Named entity recognition (NER) |
| `transformers` | ≥4.45 | DistilBERT sentiment analysis (runs locally) |
| `torch` | ≥2.4 | PyTorch backend for transformers |
| `SpeechRecognition` | ≥3.10 | High-level STT (wraps Google Web Speech API) |
| `pyaudio` | ≥0.2 | Low-level microphone access |
| `pyttsx3` | ≥2.90 | Offline text-to-speech |
| `gTTS` | ≥2.5 | Google online text-to-speech (more natural) |
| `librosa` | ≥0.10 | Audio feature extraction (pitch, energy, pauses) |
| `pandas` | ≥2.2 | Session data tables and analytics |
| `numpy` | ≥1.26 | Numerical operations on scores and audio |
| `python-dotenv` | ≥1.0 | Load API keys from `.env` file securely |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    app.py (Streamlit UI)             │
│   ┌──────────┐  ┌──────────┐  ┌─────────────────┐   │
│   │ Sidebar  │  │  Tabs    │  │  Session State  │   │
│   │ Settings │  │ Practice │  │  (persists data)│   │
│   └──────────┘  │ Chat     │  └─────────────────┘   │
│                 │ Progress │                          │
│                 └──────────┘                          │
└───────────────────────┬─────────────────────────────┘
                        │ imports
           ┌────────────┼─────────────┐
           ▼            ▼             ▼
  ┌────────────────┐ ┌──────────┐ ┌──────────────────┐
  │QuestionGenerator│ │ AICoach  │ │  AnswerEvaluator │
  │  (Gemini API)  │ │ (Gemini +│ │  (TF-IDF + NLTK  │
  │  + Fallback    │ │ LangChain│ │  + spaCy + NER)  │
  │  Question Bank │ │ + BERT)  │ └──────────────────┘
  └────────────────┘ └──────────┘
           │
  ┌────────────────┐ ┌──────────────────────────────┐
  │SessionManager  │ │       SpeechHandler           │
  │ (pandas/numpy/ │ │  (PyAudio + SpeechRecognition │
  │  sklearn trend │ │   + pyttsx3 + gTTS + librosa) │
  │  analysis)     │ └──────────────────────────────┘
  └────────────────┘
```

---

## 🎯 Interview Domains

- Software Engineering
- Data Science & ML
- System Design
- Behavioral / HR
- Product Management
- DevOps & Cloud
- Cybersecurity
- Frontend Development
- Backend Development
- Full Stack Development

---

## 💡 How Scoring Works

Each answer is scored across four dimensions:

```
Overall Score = (Relevance × 0.35) + (Completeness × 0.30)
              + (Fluency × 0.20)   + (Confidence × 0.15)
```

| Metric | Method | Max |
|---|---|---|
| **Relevance** | TF-IDF cosine similarity vs ideal answer | 100 |
| **Completeness** | % of key concepts from ideal answer found | 100 |
| **Fluency** | 100 − (5 × number of filler words) | 100 |
| **Confidence** | Based on length + absence of hedging phrases | 100 |

**Grade scale:**
- 🏆 **Excellent** → 85–100
- 👍 **Good** → 70–84
- 📘 **Average** → 55–69
- 📝 **Needs Improvement** → 0–54

---

## 🛠️ Troubleshooting

### PyAudio installation fails (Windows)
```bash
pip install pipwin
pipwin install pyaudio
```

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Streamlit won't start
Make sure your virtual environment is activated:
```bash
venv\Scripts\activate   # Windows
streamlit run app.py
```

### Gemini API errors
- Verify your key in `.env`: `GEMINI_API_KEY=AIza...`
- Check quota at [AI Studio](https://aistudio.google.com/)
- The app works in offline mode (fallback questions, no AI coaching)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Anvitha Reddy S**

---

*Built with ❤️ using Python 3.11.9, Streamlit, and Google Gemini*
