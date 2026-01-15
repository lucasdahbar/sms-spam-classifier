import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # carregar base limpa
    df = pd.read_csv("data/processed/spam_clean.csv")

    X = df["text_clean"]
    y = df["label"]

    # split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # pipeline TF-IDF + Naive Bayes
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])

    # treino
    model.fit(X_train, y_train)

    # previsões
    y_pred = model.predict(X_test)

    # métricas
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:\n")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()