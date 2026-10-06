# ─────────────────────────────────────────────────────────────────────────────
# modules/ai_coach.py
#
# PURPOSE:
#   Provide real-time AI coaching using:
#     • Google Gemini (primary – conversational feedback)
#     • LangChain      (secondary – structured coaching chain)
#     • Hugging Face Transformers (local sentiment analysis)
#
# KEY CONCEPTS COVERED:
#   • LangChain PromptTemplate + LLMChain
#   • Google Gemini streaming responses
#   • HuggingFace pipeline() for local inference
#   • Python generators (yield keyword)
# ─────────────────────────────────────────────────────────────────────────────

from typing import Generator, Optional

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM
# FakeListLLM is a LangChain testing LLM.  We use it as fallback when no
# real LLM key is available.  In production replace with ChatGoogleGenerativeAI.

import config
from modules.answer_evaluator import EvaluationResult
from modules.question_generator import InterviewQuestion


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
# LangChain PromptTemplate lets you define a string with {placeholders}
# and fill them in later with .format_prompt(**kwargs).
# This separation of template from data makes prompts reusable and testable.

COACHING_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "score", "missing_keywords", "domain"],
    template="""You are an expert interview coach specialising in {domain} roles.

A candidate was asked:
QUESTION: {question}

Their answer was:
ANSWER: {answer}

Evaluation score: {score}/100
Keywords they missed: {missing_keywords}

Provide coaching in this exact format:
1. STRENGTH: One thing they did well (1–2 sentences)
2. IMPROVEMENT: The most important thing to fix (1–2 sentences)
3. EXAMPLE: A short example or rephrasing to demonstrate the improvement
4. NEXT STEP: One actionable practice recommendation

Keep your response concise, encouraging, and specific.""",
)

FOLLOW_UP_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "follow_up"],
    template="""Based on this interview exchange:

Original Question: {question}
Candidate's Answer: {answer}
Follow-up Question: {follow_up}

Generate a brief coaching note (2–3 sentences) on how to best answer the follow-up,
given what was already said. Be specific and practical.""",
)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: AICoach
# ─────────────────────────────────────────────────────────────────────────────
class AICoach:
    """
    Delivers personalised coaching using Gemini + LangChain.

    Two modes:
      stream_coaching()    – streams tokens live (for Streamlit's st.write_stream)
      get_coaching()       – returns the full coaching text at once
    """

    def __init__(self) -> None:
        self._gemini_ready = False
        self._sentiment_pipeline = None  # Loaded lazily

        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self._model = genai.GenerativeModel(
                    model_name=config.GEMINI_MODEL,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.6,          # Slightly more deterministic for coaching
                        max_output_tokens=512,
                    ),
                )
                self._gemini_ready = True
            except Exception as e:
                print(f"[AICoach] Gemini init failed: {e}")

    # ── Streaming coaching ────────────────────────────────────────────────────
    def stream_coaching(
        self,
        question: InterviewQuestion,
        answer: str,
        evaluation: EvaluationResult,
    ) -> Generator[str, None, None]:
        """
        A GENERATOR function – note the `yield` keyword below.

        A generator is a special function that:
          • Returns a generator object (iterable) instead of a single value
          • Produces values one at a time using `yield`
          • Pauses execution between yields, resuming on the next iteration

        Streamlit's st.write_stream() calls next() on this generator to
        display each token as it arrives, creating a typewriter effect.

        Syntax:  for chunk in stream_coaching(...): ...
        """
        if not self._gemini_ready:
            # Even without Gemini we can still yield useful text
            yield from self._fallback_coaching(question, answer, evaluation)
            return
            # `return` inside a generator signals StopIteration (end of stream)

        # Build the coaching prompt using LangChain PromptTemplate
        missing_kw = ", ".join(evaluation.keywords_missing[:5]) or "none identified"
        prompt_text = COACHING_PROMPT.format(
            question        = question.text,
            answer          = answer[:500],          # Truncate very long answers
            score           = evaluation.overall_score,
            missing_keywords = missing_kw,
            domain          = question.domain,
        )

        try:
            # generate_content with stream=True returns chunks progressively.
            # Each chunk has a `.text` attribute with the next piece of text.
            response = self._model.generate_content(prompt_text)
            if response.text:
                yield response.text
        except Exception as e:
            error_message = str(e)
            
            if "429" in error_message or "quota" in error_message.lower():
                print("[AICoach] Gemini quota exceeded. Using fallback coaching.")
                yield "⚠️ Gemini quota is currently exhausted. Using offline AI coaching instead.\n\n"
                yield from self._fallback_coaching(question, answer, evaluation)
            else:
                print(f"[AICoach] Streaming error: {e}")
                yield from self._fallback_coaching(question, answer, evaluation)

    # ── Non-streaming coaching ─────────────────────────────────────────────────
    def get_coaching(
        self,
        question: InterviewQuestion,
        answer: str,
        evaluation: EvaluationResult,
    ) -> str:
        """Return the complete coaching text as a single string."""
        # Collect all generator output into a list, then join
        return "".join(self.stream_coaching(question, answer, evaluation))

    # ── Follow-up coaching ────────────────────────────────────────────────────
    def get_follow_up_coaching(
        self,
        question: InterviewQuestion,
        answer: str,
    ) -> str:
        """Generate coaching for the follow-up question."""
        if not question.follow_up:
            return ""

        if not self._gemini_ready:
            return f"Consider this follow-up: {question.follow_up}"

        prompt_text = FOLLOW_UP_PROMPT.format(
            question   = question.text,
            answer     = answer[:400],
            follow_up  = question.follow_up,
        )

        try:
            response = self._model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "quota" in error_message.lower():
                return (
                "⚠️ Gemini quota is currently exhausted. "
                "Offline coaching is being used instead.\n\n"
                f"Consider this follow-up: {question.follow_up}"
                )

        return f"Follow-up: {question.follow_up}\n(Coaching unavailable: {e})"
    # ── Sentiment analysis ─────────────────────────────────────────────────────
    def analyse_sentiment(self, text: str) -> dict:
        """
        Use a Hugging Face transformer pipeline to analyse answer sentiment.

        pipeline() is a high-level API from the `transformers` library.
        It downloads a pre-trained model (first call takes a while),
        then uses it for inference locally – no API key needed.

        Model used: distilbert-base-uncased-finetuned-sst-2-english
          - DistilBERT: smaller, faster version of BERT
          - Fine-tuned on SST-2: Stanford Sentiment Treebank
          - Output: {"label": "POSITIVE"/"NEGATIVE", "score": 0–1}
        """
        if self._sentiment_pipeline is None:
            try:
                from transformers import pipeline
                # Lazy import – we only import transformers when this method is called.
                # This avoids a heavy import at startup.
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True,    # Truncate inputs longer than model's max
                    max_length=512,
                )
            except Exception as e:
                print(f"[AICoach] Transformer pipeline init failed: {e}")
                return {"label": "UNKNOWN", "score": 0.0}

        try:
            result = self._sentiment_pipeline(text[:512])[0]
            # [0] because the pipeline returns a list; we take the first (only) result
            return result
        except Exception:
            return {"label": "UNKNOWN", "score": 0.0}

    # ── Fallback coaching (no API) ─────────────────────────────────────────────
    def _fallback_coaching(
        self,
        question: InterviewQuestion,
        answer: str,
        evaluation: EvaluationResult,
    ) -> Generator[str, None, None]:
        """Rule-based coaching when Gemini is unavailable."""
        lines = []

        score = evaluation.overall_score
        if score >= 80:
            lines.append("**STRENGTH:** Your answer demonstrates solid understanding of the topic.")
        elif score >= 55:
            lines.append("**STRENGTH:** You've covered the basics of this topic.")
        else:
            lines.append("**STRENGTH:** You've made an attempt – learning starts here!")

        if evaluation.keywords_missing:
            kw = ", ".join(evaluation.keywords_missing[:3])
            lines.append(f"\n**IMPROVEMENT:** Incorporate these key concepts: {kw}.")
        else:
            lines.append("\n**IMPROVEMENT:** Focus on structuring your answer more clearly (use STAR or P-A-R method).")

        lines.append(f"\n**NEXT STEP:** Review the ideal answer below and practise aloud.")

        for line in lines:
            yield line  # Yield each line separately to simulate streaming
            # (In the real generator, chunks arrive over time;
            #  here we emit all at once but maintain the same interface.)

    # ── AI-driven mock chat ────────────────────────────────────────────────────
    def chat(self, conversation_history: list[dict], user_message: str) -> str:
        """
        Conduct a freeform AI interview conversation.

        conversation_history is a list of {"role": "user"/"model", "parts": ["..."]}
        dicts – the format Gemini's multi-turn chat API expects.
        """
        if not self._gemini_ready:
            return "I'm sorry, the AI coach is unavailable (no API key configured)."

        try:
            # start_chat() creates a ChatSession with history
            chat_session = self._model.start_chat(history=conversation_history)
            response = chat_session.send_message(user_message)
            return response.text
        except Exception as e:
            return f"Error communicating with AI coach: {e}"
