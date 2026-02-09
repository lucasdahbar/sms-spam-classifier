# SMS Spam Classifier

This project implements a machine learning pipeline to classify SMS messages as
spam or legitimate (ham). The goal is to explore text preprocessing techniques
and evaluate supervised learning models for spam detection.

📄 Read this README in portuguese: README.pt-br.md

## Project Overview
Spam detection is a classic text classification problem in machine learning.
In this project, SMS messages are processed using natural language processing
techniques and classified using supervised algorithms.

## Dataset
The dataset consists of labeled SMS messages, where each message is categorized
as either spam or ham. The data is publicly available and commonly used for
educational purposes in machine learning and NLP tasks.

## Methodology
The project follows these main steps:
- Text preprocessing (lowercasing, tokenization, stopword removal)
- Feature extraction using TF-IDF
- Train-test split
- Training supervised classification models
- Model evaluation using standard metrics

## Models Used
- Multinomial Naive Bayes
- Support Vector Machine (Linear SVM)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Project Structure
