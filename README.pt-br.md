# Classificador de SMS Spam

Este projeto implementa um pipeline de aprendizado de máquina para classificar
mensagens SMS como spam ou legítimas (ham). O objetivo é explorar técnicas de
pré-processamento de texto e avaliar modelos supervisionados para detecção de spam.

📄 Leia este README em inglês: README.md

## Visão Geral do Projeto
A detecção de spam é um problema clássico de classificação de textos em aprendizado
de máquina. Neste projeto, mensagens SMS são processadas utilizando técnicas de
Processamento de Linguagem Natural (PLN) e classificadas por meio de algoritmos
supervisionados.

## Conjunto de Dados
O conjunto de dados é composto por mensagens SMS rotuladas como spam ou ham.
Trata-se de um dataset público, amplamente utilizado para fins educacionais em
tarefas de aprendizado de máquina e PLN.

## Metodologia
O projeto segue as seguintes etapas principais:
- Pré-processamento de texto (normalização, tokenização e remoção de stopwords)
- Extração de características utilizando TF-IDF
- Separação dos dados em treino e teste
- Treinamento de modelos de classificação supervisionada
- Avaliação dos modelos com métricas padrão

## Modelos Utilizados
- Naive Bayes Multinomial
- Support Vector Machine (SVM Linear)

## Métricas de Avaliação
- Acurácia
- Precisão
- Revocação (Recall)
- F1-score
- Matriz de Confusão

## Estrutura do Projeto

sms-spam-classifier/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
├── src/
└── README.md


## Resultados
Os modelos apresentaram desempenho superior ao acaso, evidenciando a eficácia
da vetorização de texto e do aprendizado supervisionado para o problema de
classificação de spam em mensagens SMS.

## Conclusão
Este projeto demonstra que modelos clássicos de aprendizado de máquina, aliados
a técnicas básicas de processamento de texto, são capazes de resolver de forma
efetiva o problema de classificação de spam. Além disso, o projeto serve como
uma introdução prática a pipelines de aprendizado de máquina aplicados a textos.

## Trabalhos Futuros
- Explorar outras representações de texto
- Ajuste de hiperparâmetros
- Avaliação com conjuntos de dados maiores
