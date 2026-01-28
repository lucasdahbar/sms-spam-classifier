import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC

def main():
    # carregar base limpa
    df = pd.read_csv("data/processed/spam_clean.csv")

    X = df["text_clean"]
    y = df["label"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])

    # treino
    model.fit(X_train, y_train)

    # previsões
    y_pred = model.predict(X_test)

    # métricas NB
    print("\nNaive Bayes")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:\n")
    print(confusion_matrix(y_test, y_pred))

    svm_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("svm", LinearSVC())
    ])

    # treino
    svm_model.fit(X_train, y_train)

    
    # previsões
    y_pred_svm = svm_model.predict(X_test)

    # métricas SVM
    print("\nSVM Linear")
    print("Acurácia:", accuracy_score(y_test, y_pred_svm))
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred_svm))
    print("Matriz de Confusão:\n")
    print(confusion_matrix(y_test, y_pred_svm))
    
    # relatório Naive Bayes
    report_nb = classification_report(y_test, y_pred, output_dict=True)
    df_nb = pd.DataFrame(report_nb).transpose()
    df_nb["modelo"] = "Naive Bayes"

    # relatório SVM
    report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
    df_svm = pd.DataFrame(report_svm).transpose()
    df_svm["modelo"] = "SVM Linear"

    df_compare = pd.concat([df_nb, df_svm])
    compare_spam = df_compare.loc["1"][["precision", "recall", "f1-score", "modelo"]]
    print(compare_spam)
if __name__ == "__main__":
        main()