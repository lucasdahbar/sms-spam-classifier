import re
import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = str(text).lower()

    # remove urls/emails
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)

    # remove caracteres que não ajudam muito na análise
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # normaliza espaços
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text