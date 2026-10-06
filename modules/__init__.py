# modules/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# This file makes the `modules/` directory a Python package.
#
# WHAT IS A PACKAGE?
#   In Python, a "package" is simply a folder that contains a special file
#   called __init__.py.  Without this file, Python would not recognize the
#   folder as something you can import from.
#
# HOW IMPORTS WORK WITH PACKAGES:
#   With this file present you can do:
#       from modules.question_generator import QuestionGenerator
#   Without it, the above import would raise a ModuleNotFoundError.
#
# WHAT GOES IN __init__.py?
#   It can be completely empty (like this file) or it can import and
#   re-export symbols to provide a cleaner public API.  For beginners,
#   leaving it empty is the safest and most readable approach.
# ─────────────────────────────────────────────────────────────────────────────
