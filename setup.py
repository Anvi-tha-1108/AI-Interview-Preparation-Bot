# ─────────────────────────────────────────────────────────────────────────────
# setup.py  –  One-time project setup script
#
# Run this ONCE after cloning the repo to download all required NLP models.
# Usage:  python setup.py
# ─────────────────────────────────────────────────────────────────────────────

import subprocess   # Built-in: run shell commands from Python
import sys          # Built-in: access to the Python interpreter path

def run(cmd: list[str]) -> None:
    """Run a command and print its output. Exits on failure."""
    print(f"\n▶ Running: {' '.join(cmd)}")
    # subprocess.run() runs the command in a subprocess and waits for it.
    # check=True raises CalledProcessError if the command returns non-zero exit code.
    result = subprocess.run(cmd, check=True)
    print(f"✅ Done: {' '.join(cmd)}")

def main() -> None:
    print("=" * 60)
    print("AI Interview Preparation Bot – Setup")
    print("=" * 60)

    # sys.executable is the path to the current Python interpreter.
    # Using it (instead of just "python") ensures we use the venv's Python.
    python = sys.executable

    # ── 1. Download NLTK data ──────────────────────────────────────────────────
    print("\n[1/3] Downloading NLTK data...")
    import nltk
    for package in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
        nltk.download(package, quiet=False)
    print("✅ NLTK data ready")

    # ── 2. Download spaCy model ────────────────────────────────────────────────
    print("\n[2/3] Downloading spaCy English model...")
    run([python, "-m", "spacy", "download", "en_core_web_sm"])

    # ── 3. Verify Transformers (downloads model on first use, not here) ────────
    print("\n[3/3] Verifying transformers installation...")
    try:
        import transformers
        print(f"✅ transformers {transformers.__version__} installed")
    except ImportError:
        print("❌ transformers not installed. Run: pip install transformers")

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("Next steps:")
    print("  1. Copy .env.example → .env and add your GEMINI_API_KEY")
    print("  2. Run: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    # __name__ == "__main__" is True only when this file is run directly
    # (not when it's imported by another module).
    # This is the standard Python idiom for "entry point".
    main()
