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

def main():
    # ajusta o caminho
    in_path = "data/raw/spam.csv"
    out_path = "data/processed/spam_clean.csv"

    df = pd.read_csv(in_path, encoding="latin-1")

    # mantém só label + texto
    # dataset clássico: v1=label, v2=text
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})

    # remove nulos / vazios
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().ne("")].copy()

    # remove duplicatas 
    df = df.drop_duplicates(subset=["label", "text"])

    # mapeia labels 
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # remove linhas que ficaram sem label (por segurança)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # limpa texto
    df["text_clean"] = df["text"].apply(clean_text)

    # remove mensagens que viraram vazias após limpeza
    df = df[df["text_clean"].str.strip().ne("")].copy()

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Base limpa salva em: {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()