# ─────────────────────────────────────────────────────────────────────────────
# modules/answer_evaluator.py
#
# PURPOSE:
#   Score a candidate's answer against an ideal answer using:
#     1. TF-IDF cosine similarity  (scikit-learn)
#     2. NLTK-based NLP analysis   (keyword extraction, filler words)
#     3. spaCy named-entity analysis
#     4. Transformers sentiment     (optional, heavy – loads lazily)
#
# KEY CONCEPTS COVERED:
#   • TF-IDF (Term Frequency–Inverse Document Frequency)
#   • Cosine Similarity
#   • NLTK tokenisation & stopword removal
#   • spaCy named entity recognition (NER)
#   • Python @property decorator
#   • Lazy initialisation
# ─────────────────────────────────────────────────────────────────────────────

import re                                # Built-in: regular expressions for text cleaning
from dataclasses import dataclass, field
from typing import Optional

import nltk                              # Natural Language Toolkit
import spacy                             # Industrial-strength NLP library
from nltk.corpus import stopwords        # Common words to ignore (the, is, a …)
from nltk.tokenize import word_tokenize  # Splits text into tokens (words)
from sklearn.feature_extraction.text import TfidfVectorizer   # TF-IDF engine
from sklearn.metrics.pairwise import cosine_similarity        # Similarity metric

import config

# ── Download NLTK data (only needed once; safe to call every time) ────────────
# nltk.download() checks if the data is already on disk; if yes, it skips.
# 'quiet=True' suppresses the download progress bar.
nltk.download("punkt",        quiet=True)   # Tokeniser rules
nltk.download("punkt_tab",   quiet=True)   # Updated tokeniser tables
nltk.download("stopwords",   quiet=True)   # English stop-word list
nltk.download("averaged_perceptron_tagger", quiet=True)  # POS tagger


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS: EvaluationResult
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    """Holds every metric produced by the evaluator for one answer."""

    overall_score: float        # 0–100 composite score
    similarity_score: float     # TF-IDF cosine similarity (0–1) × 100
    completeness_score: float   # Keyword coverage score (0–100)
    fluency_score: float        # Penalised for filler words (0–100)
    confidence_score: float     # Estimated from sentence length & structure
    keywords_used: list[str]    = field(default_factory=list)   # Good keywords found
    keywords_missing: list[str] = field(default_factory=list)   # Expected but absent
    filler_words_found: list[str] = field(default_factory=list) # Disfluencies detected
    entities_found: list[str]   = field(default_factory=list)   # Named entities (spaCy)
    word_count: int             = 0
    sentence_count: int         = 0
    feedback: str               = ""      # Human-readable summary
    suggestions: list[str]      = field(default_factory=list)   # Improvement tips

    @property
    def grade(self) -> str:
        """
        A @property turns a method into an attribute you can read without ().
        Instead of  result.grade()  you write  result.grade
        The property decorator makes it feel like a stored value even though
        it's computed fresh each time.
        """
        if self.overall_score >= 85:
            return "Excellent 🏆"
        elif self.overall_score >= 70:
            return "Good 👍"
        elif self.overall_score >= 55:
            return "Average 📘"
        else:
            return "Needs Improvement 📝"


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: AnswerEvaluator
# ─────────────────────────────────────────────────────────────────────────────
class AnswerEvaluator:
    """
    Evaluates interview answers using NLP techniques.

    Design decision: spaCy model is loaded lazily (only when first needed)
    because it is ~50 MB and slows startup.  This is called "lazy initialisation".
    """

    # Class-level variable (shared across ALL instances of AnswerEvaluator).
    # None means "not loaded yet".
    _nlp_model = None   # Will hold the spaCy model once loaded

    def __init__(self) -> None:
        self._stop_words: set[str] = set(stopwords.words("english"))
        # set() is used because membership testing (`word in stop_words`) is
        # O(1) for sets vs O(n) for lists.

        self._vectorizer = TfidfVectorizer(
            stop_words="english",   # TF-IDF also removes English stop-words
            ngram_range=(1, 2),     # Consider single words AND 2-word phrases
            max_features=5000,      # Limit vocabulary size for efficiency
        )

    # ── Lazy-load spaCy ──────────────────────────────────────────────────────
    @classmethod
    def _get_nlp(cls):
        """
        @classmethod receives the CLASS as first arg (cls) instead of an instance.
        Here we use it to implement a "singleton" pattern:
        load the model once, store it on the class, reuse it forever.
        """
        if cls._nlp_model is None:
            try:
                cls._nlp_model = spacy.load("en_core_web_sm")
                # "en_core_web_sm" is spaCy's small English model.
                # Install it with:  python -m spacy download en_core_web_sm
            except OSError:
                # OSError is raised if the model file is not found.
                # We return None and skip NER analysis gracefully.
                print("[AnswerEvaluator] spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return cls._nlp_model

    # ── Main public method ────────────────────────────────────────────────────
    def evaluate(
        self,
        answer: str,
        ideal_answer: str,
        question_text: str = "",
    ) -> EvaluationResult:
        """
        Score `answer` against `ideal_answer`.

        Parameters
        ----------
        answer       : Candidate's spoken/typed answer
        ideal_answer : Reference answer from the question bank
        question_text: The question (used for keyword extraction)

        Returns
        -------
        EvaluationResult
        """
        # ── Pre-process ───────────────────────────────────────────────────────
        answer_clean  = self._clean_text(answer)
        ideal_clean   = self._clean_text(ideal_answer)

        # Guard: if answer is empty, return zero score immediately
        if not answer_clean:
            return EvaluationResult(
                overall_score=0, similarity_score=0, completeness_score=0,
                fluency_score=0, confidence_score=0,
                feedback="No answer provided.",
                suggestions=["Please provide an answer before submitting."],
            )

        # ── Individual scores ─────────────────────────────────────────────────
        similarity    = self._tfidf_similarity(answer_clean, ideal_clean)
        completeness, kw_used, kw_missing = self._keyword_analysis(answer_clean, ideal_clean)
        fluency, fillers               = self._fluency_analysis(answer)
        confidence                     = self._confidence_analysis(answer)
        entities                       = self._entity_analysis(answer)
        word_count, sentence_count     = self._count_stats(answer)

        # ── Weighted composite score ──────────────────────────────────────────
        # Each metric contributes a different weight based on importance.
        # The weights sum to 1.0.
        overall = (
            similarity   * 0.35 +   # Does the answer cover the same ground?
            completeness * 0.30 +   # Are key concepts mentioned?
            fluency      * 0.20 +   # Is speech clear without fillers?
            confidence   * 0.15     # Does it sound confident and structured?
        )
        # Clamp to [0, 100] range with min/max
        overall = max(0.0, min(100.0, overall))

        # ── Build feedback text ───────────────────────────────────────────────
        feedback, suggestions = self._build_feedback(
            similarity, completeness, fluency, confidence,
            kw_missing, fillers, word_count
        )

        return EvaluationResult(
            overall_score    = round(overall, 1),
            similarity_score = round(similarity, 1),
            completeness_score = round(completeness, 1),
            fluency_score    = round(fluency, 1),
            confidence_score = round(confidence, 1),
            keywords_used    = kw_used,
            keywords_missing = kw_missing,
            filler_words_found = fillers,
            entities_found   = entities,
            word_count       = word_count,
            sentence_count   = sentence_count,
            feedback         = feedback,
            suggestions      = suggestions,
        )

    # ── Helper: clean text ────────────────────────────────────────────────────
    def _clean_text(self, text: str) -> str:
        """
        Lowercase and remove punctuation.
        re.sub(pattern, replacement, string) replaces all regex matches.
        [^a-z0-9\s] matches any character that is NOT a letter, digit, or space.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # \s+ matches one or more whitespace characters.
        # .strip() removes leading/trailing whitespace.
        return text

    # ── Helper: TF-IDF cosine similarity ─────────────────────────────────────
    def _tfidf_similarity(self, answer: str, ideal: str) -> float:
        """
        TF-IDF (Term Frequency–Inverse Document Frequency):
          - TF:  how often a word appears in THIS document
          - IDF: how rare the word is across ALL documents
          - TF-IDF: words that are frequent here but rare elsewhere get high scores
                    common words like "the" get near-zero weight

        Cosine Similarity:
          Treats each document as a vector in high-dimensional space.
          Measures the angle between the two vectors.
          0 = completely different,  1 = identical content.

        We scale the result to 0–100.
        """
        try:
            # fit_transform() learns the vocabulary from BOTH texts and
            # returns a sparse matrix (rows = documents, cols = vocabulary words).
            tfidf_matrix = self._vectorizer.fit_transform([answer, ideal])
            # cosine_similarity returns a 2×2 matrix; [0, 1] is answer vs ideal.
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(score) * 100
        except Exception:
            return 0.0

    # ── Helper: keyword analysis ──────────────────────────────────────────────
    def _keyword_analysis(
        self, answer: str, ideal: str
    ) -> tuple[float, list[str], list[str]]:
        """
        Extract important words from the ideal answer (after removing stop-words),
        then check how many appear in the candidate's answer.

        Returns
        -------
        (score_0_to_100, keywords_used, keywords_missing)

        tuple is an immutable sequence of fixed length.
        tuple[float, list[str], list[str]] means "a 3-element tuple of those types".
        """
        # Tokenise (split into words) using NLTK
        ideal_tokens  = word_tokenize(ideal)
        answer_tokens = set(word_tokenize(answer))
        # Using set for answer tokens → O(1) membership test

        # Filter: keep only meaningful words (not stop-words, not single chars)
        keywords = [
            w for w in ideal_tokens
            if w not in self._stop_words and len(w) > 2
        ]
        # List comprehension: build a list by filtering ideal_tokens.

        if not keywords:
            return (0.0, [], [])
            # Early return with a 3-element tuple

        used    = [kw for kw in keywords if kw in answer_tokens]
        missing = [kw for kw in keywords if kw not in answer_tokens]

        score = (len(used) / len(keywords)) * 100
        return (score, used[:10], missing[:10])  # Cap lists at 10 items

    # ── Helper: fluency analysis ──────────────────────────────────────────────
    def _fluency_analysis(self, answer: str) -> tuple[float, list[str]]:
        """
        Detect filler words and compute a fluency penalty.
        A perfect answer (zero fillers) scores 100.
        Each filler found deducts points (capped at 0).
        """
        answer_lower = answer.lower()
        found_fillers: list[str] = []

        for filler in config.FILLER_WORDS:
            # re.findall returns a list of all matches.
            # r"\b" is a word boundary – prevents "like" matching "dislike".
            # re.IGNORECASE makes it case-insensitive.
            matches = re.findall(rf"\b{re.escape(filler)}\b", answer_lower, re.IGNORECASE)
            if matches:
                found_fillers.append(filler)

        # Penalty: 5 points per filler word, capped at 0
        penalty = len(found_fillers) * 5
        score = max(0.0, 100.0 - penalty)
        return (score, found_fillers)

    # ── Helper: confidence analysis ───────────────────────────────────────────
    def _confidence_analysis(self, answer: str) -> float:
        """
        Heuristic confidence score based on:
          - Answer length (longer = more confident, up to a point)
          - Use of hedging language ("I think", "maybe", "I'm not sure")
          - Sentence structure variety
        """
        words = answer.split()
        word_count = len(words)

        # Length score: 50–150 words is ideal (score = 100); outside that, penalise
        if word_count < config.MIN_ANSWER_WORDS:
            length_score = (word_count / config.MIN_ANSWER_WORDS) * 50
        elif word_count <= 150:
            # Linear interpolation between MIN_ANSWER_WORDS and 150
            length_score = 50 + (word_count - config.MIN_ANSWER_WORDS) / (150 - config.MIN_ANSWER_WORDS) * 50
        else:
            length_score = 100.0

        # Hedging penalty
        hedges = ["i think", "i'm not sure", "maybe", "perhaps", "i guess", "i believe"]
        hedge_count = sum(1 for h in hedges if h in answer.lower())
        hedge_penalty = hedge_count * 5

        score = max(0.0, length_score - hedge_penalty)
        return min(100.0, score)

    # ── Helper: named entity recognition ─────────────────────────────────────
    def _entity_analysis(self, answer: str) -> list[str]:
        """
        Use spaCy to find named entities (people, organisations, technologies).
        Named entities show domain knowledge and specificity.
        """
        nlp = self._get_nlp()
        if nlp is None:
            return []

        doc = nlp(answer)
        # doc.ents is a tuple of Span objects (each with .text and .label_)
        # PERSON, ORG, PRODUCT, GPE are spaCy entity types
        relevant_labels = {"ORG", "PRODUCT", "GPE", "LANGUAGE", "WORK_OF_ART"}
        entities = [
            ent.text for ent in doc.ents
            if ent.label_ in relevant_labels
        ]
        return list(set(entities))  # Remove duplicates with set(), convert back to list

    # ── Helper: count word and sentence stats ─────────────────────────────────
    def _count_stats(self, answer: str) -> tuple[int, int]:
        """Count words and sentences in the raw answer text."""
        words     = len(answer.split())
        # Simple sentence split: count occurrences of '.', '!', '?'
        sentences = len(re.findall(r"[.!?]+", answer))
        sentences = max(1, sentences)  # At least 1 to avoid division by zero
        return (words, sentences)

    # ── Helper: build human-readable feedback ─────────────────────────────────
    def _build_feedback(
        self,
        similarity: float,
        completeness: float,
        fluency: float,
        confidence: float,
        missing_kw: list[str],
        fillers: list[str],
        word_count: int,
    ) -> tuple[str, list[str]]:
        """Generate a plain-English feedback summary and improvement list."""
        parts: list[str] = []      # Will be joined into a paragraph
        suggestions: list[str] = []

        # Similarity feedback
        if similarity >= 70:
            parts.append("Your answer aligns well with the expected response.")
        elif similarity >= 40:
            parts.append("Your answer partially covers the topic.")
            suggestions.append("Revisit the core concepts and add more depth.")
        else:
            parts.append("Your answer diverges significantly from the ideal response.")
            suggestions.append("Study the model answer and try again.")

        # Completeness feedback
        if completeness < 50 and missing_kw:
            kw_str = ", ".join(missing_kw[:5])
            suggestions.append(f"Consider including these key terms: {kw_str}.")

        # Fluency feedback
        if fillers:
            suggestions.append(
                f"Reduce filler words: {', '.join(fillers)}. Practice pausing instead."
            )

        # Length feedback
        if word_count < config.MIN_ANSWER_WORDS:
            parts.append("Your answer is too brief.")
            suggestions.append("Aim for at least 50 words; elaborate with examples.")
        elif word_count > 300:
            parts.append("Your answer is quite long; try to be more concise.")
            suggestions.append("Keep answers under 2–3 minutes when spoken.")

        # Confidence feedback
        if confidence < 50:
            suggestions.append("Avoid hedging phrases like 'I think' or 'I'm not sure'. Be assertive.")

        feedback = " ".join(parts) if parts else "Answer evaluated."
        return (feedback, suggestions)
