# ─────────────────────────────────────────────────────────────────────────────
# modules/question_generator.py
#
# PURPOSE:
#   Generate interview questions using the Google Gemini API (google-genai).
#   Also provides a bank of hand-crafted fallback questions for offline use.
#
# KEY CONCEPTS COVERED:
#   • google-genai client setup
#   • Prompt engineering (system + user messages)
#   • Python dataclasses
#   • Exception handling
#   • Type hints
# ─────────────────────────────────────────────────────────────────────────────

import json                              # Built-in: parse JSON strings into Python dicts
import random                            # Built-in: shuffle / sample lists
from dataclasses import dataclass, field # Built-in: create lightweight data containers
from typing import Optional              # Built-in: annotate "value or None"

import google.generativeai as genai      # Google Gemini SDK (pip: google-genai)

import config                            # Our config.py in the project root


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS: InterviewQuestion
# ─────────────────────────────────────────────────────────────────────────────
# A dataclass is a Python 3.7+ feature that auto-generates __init__, __repr__,
# and __eq__ methods based on the class attributes you declare with type hints.
#
# WITHOUT dataclass you'd write:
#   class InterviewQuestion:
#       def __init__(self, text, domain, difficulty, category, ideal_answer, hints):
#           self.text = text
#           ...  (8 more lines)
#
# WITH @dataclass you just declare the fields – Python does the rest.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InterviewQuestion:
    """Represents a single interview question with metadata."""

    text: str                           # The actual question text
    domain: str                         # e.g. "Software Engineering"
    difficulty: str                     # "Beginner" | "Intermediate" | "Advanced"
    category: str                       # e.g. "Technical" | "Behavioral" | "Situational"
    ideal_answer: str                   # A model answer for scoring reference
    hints: list[str] = field(default_factory=list)
    # ↑ field(default_factory=list) means "default value is an empty list []"
    #   We MUST use default_factory for mutable defaults (lists, dicts) –
    #   using `hints: list = []` would share the SAME list across all instances!
    follow_up: Optional[str] = None
    # Optional[str] means the value is either a str or None.
    # In Python 3.10+ you can write `str | None` instead.


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK QUESTION BANK  (used when API is unavailable / rate-limited)
# ─────────────────────────────────────────────────────────────────────────────
# This is a dictionary where:
#   key   = domain name  (str)
#   value = list of InterviewQuestion objects
#
# dict[str, list[InterviewQuestion]] is a generic type hint:
#   "a dict whose keys are strings and values are lists of InterviewQuestion"

FALLBACK_QUESTIONS: dict[str, list[InterviewQuestion]] = {
    "Software Engineering": [
        InterviewQuestion(
            text="What is the difference between a process and a thread?",
            domain="Software Engineering",
            difficulty="Intermediate",
            category="Technical",
            ideal_answer=(
                "A process is an independent program with its own memory space. "
                "A thread is a lightweight unit of execution within a process. "
                "Threads share memory; processes are isolated. "
                "Context switching between threads is faster than between processes."
            ),
            hints=["Think about memory isolation", "Consider context-switching cost"],
            follow_up="When would you prefer multi-processing over multi-threading?",
        ),
        InterviewQuestion(
            text="Explain the SOLID principles with examples.",
            domain="Software Engineering",
            difficulty="Advanced",
            category="Technical",
            ideal_answer=(
                "S – Single Responsibility: one class, one reason to change. "
                "O – Open/Closed: open for extension, closed for modification. "
                "L – Liskov Substitution: subtypes must be substitutable. "
                "I – Interface Segregation: prefer many small interfaces. "
                "D – Dependency Inversion: depend on abstractions, not concretions."
            ),
            hints=["Start with Single Responsibility", "Give a code example for each"],
        ),
        InterviewQuestion(
            text="What is Big-O notation and why does it matter?",
            domain="Software Engineering",
            difficulty="Beginner",
            category="Technical",
            ideal_answer=(
                "Big-O describes how an algorithm's runtime (or space) scales "
                "with input size n.  O(1) is constant, O(n) is linear, O(n²) is "
                "quadratic.  It matters because a poorly chosen algorithm can make "
                "a program 1000× slower on large data sets."
            ),
            hints=["Give concrete examples like O(1), O(n), O(log n)"],
        ),
    ],
    "Data Science & ML": [
        InterviewQuestion(
            text="What is the bias-variance tradeoff?",
            domain="Data Science & ML",
            difficulty="Intermediate",
            category="Technical",
            ideal_answer=(
                "Bias is error from wrong assumptions (underfitting). "
                "Variance is error from sensitivity to training data (overfitting). "
                "Increasing model complexity reduces bias but raises variance. "
                "The goal is to find the sweet spot that minimises total error."
            ),
            hints=["Think about underfitting vs overfitting", "Mention regularisation"],
            follow_up="How does cross-validation help manage this tradeoff?",
        ),
        InterviewQuestion(
            text="Explain how gradient descent works.",
            domain="Data Science & ML",
            difficulty="Intermediate",
            category="Technical",
            ideal_answer=(
                "Gradient descent minimises a loss function by iteratively moving "
                "parameters in the opposite direction of the gradient. "
                "The learning rate controls step size. "
                "Variants: batch GD, stochastic GD (SGD), mini-batch GD."
            ),
            hints=["Use a hill-climbing analogy", "Mention learning rate"],
        ),
    ],
    "Behavioral / HR": [
        InterviewQuestion(
            text="Tell me about a time you handled a conflict with a teammate.",
            domain="Behavioral / HR",
            difficulty="Intermediate",
            category="Behavioral",
            ideal_answer=(
                "Use the STAR method: Situation, Task, Action, Result. "
                "Describe a real scenario, your specific actions (active listening, "
                "compromise, escalation if needed), and a positive outcome. "
                "Show empathy and communication skills."
            ),
            hints=["Use the STAR framework", "Focus on YOUR actions, not blaming others"],
        ),
        InterviewQuestion(
            text="Where do you see yourself in 5 years?",
            domain="Behavioral / HR",
            difficulty="Beginner",
            category="Behavioral",
            ideal_answer=(
                "Align your answer with the role and company growth opportunities. "
                "Show ambition (leadership, expertise) while being realistic. "
                "Mention how this position is a stepping stone."
            ),
            hints=["Research the company's growth path", "Be honest but strategic"],
        ),
    ],
    "System Design": [
        InterviewQuestion(
            text="Design a URL shortener like bit.ly.",
            domain="System Design",
            difficulty="Advanced",
            category="Technical",
            ideal_answer=(
                "Key components: API gateway, encoding service (Base62), "
                "NoSQL DB (Cassandra/DynamoDB) for URL mapping, CDN for redirect speed, "
                "caching layer (Redis) for hot URLs, analytics service. "
                "Discuss CAP theorem, horizontal scaling, and hash collision handling."
            ),
            hints=["Start with requirements (scale, latency)", "Discuss encoding strategy"],
            follow_up="How would you handle 10 billion URLs?",
        ),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: QuestionGenerator
# ─────────────────────────────────────────────────────────────────────────────
class QuestionGenerator:
    """
    Generates interview questions via Gemini API with fallback to a local bank.

    Usage:
        gen = QuestionGenerator()
        questions = gen.generate(domain="Software Engineering",
                                 difficulty="Intermediate",
                                 count=5)
    """

    def __init__(self) -> None:
        """
        __init__ is the constructor – called automatically when you do
        QuestionGenerator().  `self` refers to the newly created object.
        `-> None` is a return-type hint (constructors never return a value).
        """
        self._client_ready: bool = False  # Will be True once Gemini is configured

        # Try to initialise the Gemini client
        if config.GEMINI_API_KEY:
            try:
                # genai.configure sets the API key globally for the genai library.
                # Any subsequent genai.GenerativeModel() calls will use this key.
                genai.configure(api_key=config.GEMINI_API_KEY)

                # Create a GenerativeModel instance.
                # GenerationConfig controls output behaviour (temperature, max tokens).
                self._model = genai.GenerativeModel(
                    model_name=config.GEMINI_MODEL,
                    generation_config=genai.types.GenerationConfig(
                        temperature=config.GEMINI_TEMPERATURE,
                        max_output_tokens=config.GEMINI_MAX_TOKENS,
                    ),
                )
                self._client_ready = True
            except Exception as e:
                # Broad `except Exception` catches any error during setup.
                # We don't crash – we just note that the API isn't available.
                print(f"[QuestionGenerator] Gemini init failed: {e}")
        else:
            print("[QuestionGenerator] No API key – using fallback questions only.")

    # ── Public method ─────────────────────────────────────────────────────────
    def generate(
        self,
        domain: str,
        difficulty: str,
        count: int = 5,
        topic: str = "",
    ) -> list[InterviewQuestion]:
        """
        Return `count` questions for the given domain / difficulty.

        Parameters
        ----------
        domain      : Interview domain (e.g. "Software Engineering")
        difficulty  : "Beginner" | "Intermediate" | "Advanced"
        count       : How many questions to return (default 5)
        topic       : Optional sub-topic to focus on (e.g. "linked lists")

        Returns
        -------
        list[InterviewQuestion]
        """
        # Try the Gemini API first
        if self._client_ready:
            try:
                return self._generate_via_api(domain, difficulty, count, topic)
            except Exception as e:
                print(f"[QuestionGenerator] API call failed: {e} – using fallback.")

        # Fall back to the local question bank
        return self._get_fallback(domain, difficulty, count)

    # ── Private helper: API path ──────────────────────────────────────────────
    def _generate_via_api(
        self, domain: str, difficulty: str, count: int, topic: str
    ) -> list[InterviewQuestion]:
        """
        Ask Gemini to generate questions and parse its JSON response.

        Method name starts with `_` by convention → it's "private" (internal use).
        Python does NOT enforce privacy; _ is just a signal to other developers.
        """
        # ── Prompt Engineering ────────────────────────────────────────────────
        # We ask the model to return a JSON array so we can parse it reliably.
        # A well-structured prompt is the most important part of working with LLMs.
        topic_clause = f" focusing on {topic}" if topic else ""
        prompt = f"""You are an expert technical interviewer.
Generate exactly {count} {difficulty}-level interview questions for the domain: {domain}{topic_clause}.

Return ONLY a valid JSON array.  Each element must have these keys:
  - "text"         : the interview question (string)
  - "category"     : one of "Technical", "Behavioral", "Situational" (string)
  - "ideal_answer" : a concise model answer (string, max 150 words)
  - "hints"        : list of 2–3 short hints (list of strings)
  - "follow_up"    : a follow-up question (string, can be null)

Example format:
[
  {{
    "text": "...",
    "category": "Technical",
    "ideal_answer": "...",
    "hints": ["hint1", "hint2"],
    "follow_up": "..."
  }}
]

Return ONLY the JSON array – no markdown, no explanation."""

        # ── API Call ──────────────────────────────────────────────────────────
        response = self._model.generate_content(prompt)
        raw_text: str = response.text.strip()

        # ── Parse JSON ────────────────────────────────────────────────────────
        # Sometimes the model wraps JSON in ```json ... ``` – strip that.
        if raw_text.startswith("```"):
            # Split on newlines, skip the first and last fence lines
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1])

        data: list[dict] = json.loads(raw_text)
        # json.loads converts a JSON string into a Python object.
        # list[dict] means we expect a list of dictionaries.

        # ── Convert dicts → InterviewQuestion objects ─────────────────────────
        questions: list[InterviewQuestion] = []
        for item in data:
            # dict.get(key, default) – safer than dict[key] (won't raise KeyError)
            q = InterviewQuestion(
                text=item.get("text", ""),
                domain=domain,
                difficulty=difficulty,
                category=item.get("category", "Technical"),
                ideal_answer=item.get("ideal_answer", ""),
                hints=item.get("hints", []),
                follow_up=item.get("follow_up"),
            )
            questions.append(q)

        return questions

    # ── Private helper: Fallback path ─────────────────────────────────────────
    def _get_fallback(
        self, domain: str, difficulty: str, count: int
    ) -> list[InterviewQuestion]:
        """Return random questions from the local bank, filtered by domain."""
        # dict.get(key, default) – if the domain isn't in our bank, return []
        pool: list[InterviewQuestion] = FALLBACK_QUESTIONS.get(domain, [])

        # Also include behavioral questions regardless of domain
        if domain != "Behavioral / HR":
            pool = pool + FALLBACK_QUESTIONS.get("Behavioral / HR", [])

        if not pool:
            # Flatten ALL questions from all domains as a last resort
            pool = [q for questions in FALLBACK_QUESTIONS.values() for q in questions]
            # ↑ List comprehension with a nested loop:
            #   for each key in FALLBACK_QUESTIONS, get its list, then for each
            #   question in that list, add it to the new flat list.

        # random.sample(population, k) returns k unique elements from population.
        # min(count, len(pool)) prevents asking for more items than available.
        return random.sample(pool, min(count, len(pool)))
