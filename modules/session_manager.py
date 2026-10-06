# ─────────────────────────────────────────────────────────────────────────────
# modules/session_manager.py
#
# PURPOSE:
#   Track the user's interview session state:
#     • Questions asked / answered
#     • Scores over time
#     • Session history (for resume / review)
#     • Progress analytics (pandas + scikit-learn)
#
# KEY CONCEPTS COVERED:
#   • Streamlit session_state (persistent variables across reruns)
#   • pandas DataFrame for tabular data
#   • numpy for numerical statistics
#   • sklearn LinearRegression for trend analysis
#   • datetime for timestamping
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations  # Allows forward references in type hints

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import uuid                          # Built-in: generate unique IDs

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from modules.answer_evaluator import EvaluationResult
from modules.question_generator import InterviewQuestion


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS: SessionEntry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SessionEntry:
    """One Q&A pair with its evaluation result."""

    question_text:   str
    domain:          str
    difficulty:      str
    user_answer:     str
    overall_score:   float
    similarity_score: float
    fluency_score:   float
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat())
    # default_factory=lambda: ...  is a "zero-argument function" (lambda with no args)
    # that is called each time a new SessionEntry is created.
    # datetime.now().isoformat() → "2024-10-06T10:30:00.123456" (ISO 8601 format)

    audio_quality:   dict = field(default_factory=dict)
    coaching_notes:  str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: SessionManager
# ─────────────────────────────────────────────────────────────────────────────
class SessionManager:
    """
    Manages interview session state using Streamlit's st.session_state.

    st.session_state is a special Streamlit dictionary that persists across
    script reruns (Streamlit re-executes the whole script on every user interaction).
    Without session_state, all variables would reset on every click.

    We use the class as a thin wrapper around session_state to keep
    session logic in one place.
    """

    # ── Keys used in st.session_state ──────────────────────────────────────────
    # Using constants avoids typos like "session_entires" vs "session_entries"
    _KEY_ENTRIES       = "session_entries"
    _KEY_CURRENT_IDX   = "current_question_idx"
    _KEY_QUESTIONS     = "session_questions"
    _KEY_SESSION_ID    = "session_id"
    _KEY_DOMAIN        = "selected_domain"
    _KEY_DIFFICULTY    = "selected_difficulty"
    _KEY_CHAT_HISTORY  = "chat_history"
    _KEY_STARTED       = "session_started"

    def __init__(self) -> None:
        """Initialise session_state keys if they don't exist yet."""
        self._init_state()

    def _init_state(self) -> None:
        """
        setdefault() on a dict sets a key ONLY if it is not already present.
        This is idempotent – safe to call on every Streamlit rerun.
        """
        defaults = {
            self._KEY_ENTRIES:      [],     # list[SessionEntry]
            self._KEY_CURRENT_IDX:  0,      # index into current question list
            self._KEY_QUESTIONS:    [],     # list[InterviewQuestion]
            self._KEY_SESSION_ID:   str(uuid.uuid4()),  # unique session identifier
            self._KEY_DOMAIN:       "",
            self._KEY_DIFFICULTY:   "",
            self._KEY_CHAT_HISTORY: [],
            self._KEY_STARTED:      False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # ── Properties (read-only views into session_state) ───────────────────────
    @property
    def entries(self) -> list[SessionEntry]:
        return st.session_state[self._KEY_ENTRIES]

    @property
    def current_question_idx(self) -> int:
        return st.session_state[self._KEY_CURRENT_IDX]

    @property
    def questions(self) -> list[InterviewQuestion]:
        return st.session_state[self._KEY_QUESTIONS]

    @property
    def session_id(self) -> str:
        return st.session_state[self._KEY_SESSION_ID]

    @property
    def is_started(self) -> bool:
        return st.session_state[self._KEY_STARTED]

    @property
    def current_question(self) -> Optional[InterviewQuestion]:
        """Returns the active question or None if session is over."""
        questions = self.questions
        idx = self.current_question_idx
        if questions and idx < len(questions):
            return questions[idx]
        return None

    # ── Setters ────────────────────────────────────────────────────────────────
    def start_session(
        self,
        questions: list[InterviewQuestion],
        domain: str,
        difficulty: str,
    ) -> None:
        """Initialise a new practice session with the given questions."""
        st.session_state[self._KEY_QUESTIONS]   = questions
        st.session_state[self._KEY_ENTRIES]     = []
        st.session_state[self._KEY_CURRENT_IDX] = 0
        st.session_state[self._KEY_DOMAIN]      = domain
        st.session_state[self._KEY_DIFFICULTY]  = difficulty
        st.session_state[self._KEY_STARTED]     = True
        st.session_state[self._KEY_SESSION_ID]  = str(uuid.uuid4())

    def record_answer(
        self,
        question: InterviewQuestion,
        answer: str,
        evaluation: EvaluationResult,
        audio_quality: dict = None,
        coaching_notes: str = "",
    ) -> None:
        """Save a completed Q&A pair to the session history."""
        entry = SessionEntry(
            question_text    = question.text,
            domain           = question.domain,
            difficulty       = question.difficulty,
            user_answer      = answer,
            overall_score    = evaluation.overall_score,
            similarity_score = evaluation.similarity_score,
            fluency_score    = evaluation.fluency_score,
            audio_quality    = audio_quality or {},
            coaching_notes   = coaching_notes,
        )
        st.session_state[self._KEY_ENTRIES].append(entry)
        # Advance to next question
        st.session_state[self._KEY_CURRENT_IDX] += 1

    def add_chat_message(self, role: str, content: str) -> None:
        """Append a message to the chat history."""
        st.session_state[self._KEY_CHAT_HISTORY].append({
            "role": role,       # "user" or "assistant"
            "content": content,
        })

    def reset(self) -> None:
        """Clear all session data (start fresh)."""
        for key in [
            self._KEY_ENTRIES, self._KEY_QUESTIONS, self._KEY_CHAT_HISTORY
        ]:
            st.session_state[key] = []
        st.session_state[self._KEY_CURRENT_IDX] = 0
        st.session_state[self._KEY_STARTED]     = False

    # ── Analytics ──────────────────────────────────────────────────────────────
    def get_analytics_df(self) -> pd.DataFrame:
        """
        Convert session entries into a pandas DataFrame for analysis.

        pd.DataFrame can be constructed from a list of dicts.
        dataclasses.asdict() converts a dataclass instance → a plain dict.
        """
        if not self.entries:
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                "question_text", "domain", "difficulty",
                "overall_score", "similarity_score", "fluency_score", "timestamp"
            ])

        records = [asdict(e) for e in self.entries]
        # asdict() recursively converts nested dataclasses too.

        df = pd.DataFrame(records)
        # Parse timestamp strings back into proper datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_summary_stats(self) -> dict:
        """
        Compute summary statistics using pandas and numpy.

        Returns a dict with mean, max, min, and trend of scores.
        """
        df = self.get_analytics_df()
        if df.empty:
            return {}

        scores = df["overall_score"].values   # numpy array (via pandas)
        # .values converts a pandas Series → numpy ndarray
        # This lets us use numpy functions directly.

        stats = {
            "mean_score":   float(np.mean(scores)),
            "max_score":    float(np.max(scores)),
            "min_score":    float(np.min(scores)),
            "std_score":    float(np.std(scores)),
            "total_questions": len(scores),
            "by_domain":    df.groupby("domain")["overall_score"].mean().to_dict(),
        }

        # ── Score trend using sklearn LinearRegression ─────────────────────────
        # We fit a line through (question_number, score) points.
        # The slope tells us if the user is improving (positive) or not.
        if len(scores) >= 3:
            X = np.arange(len(scores)).reshape(-1, 1)
            # np.arange(n) → [0, 1, 2, ..., n-1]
            # .reshape(-1, 1) reshapes to a column vector: [[0], [1], [2], ...]
            # sklearn expects X to be 2D (n_samples, n_features)

            y = scores
            reg = LinearRegression()
            reg.fit(X, y)
            # fit() learns the best-fit line parameters (slope and intercept)
            stats["score_trend"] = float(reg.coef_[0])
            # coef_[0] is the slope: positive = improving, negative = declining
            stats["trend_label"] = (
                "📈 Improving" if reg.coef_[0] > 0.5
                else "📉 Declining" if reg.coef_[0] < -0.5
                else "➡️ Stable"
            )

        return stats

    # ── Export ─────────────────────────────────────────────────────────────────
    def export_json(self) -> str:
        """Export session data as a JSON string for download."""
        data = {
            "session_id": self.session_id,
            "entries": [asdict(e) for e in self.entries],
            "stats": self.get_summary_stats(),
        }
        return json.dumps(data, indent=2, default=str)
        # default=str: if a value can't be serialized (e.g. datetime), convert it to str
