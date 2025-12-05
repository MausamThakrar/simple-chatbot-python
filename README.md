# Simple Intent-Based Chatbot in Python

This project implements a small **console-based chatbot** using
a basic Natural Language Processing (NLP) pipeline with scikit-learn.

The chatbot recognises simple **intents** such as:

- greeting
- goodbye
- thanks
- asking for the bot's name
- weather-related questions (very simple)
- small talk

and responds with predefined messages based on the predicted intent.

## Main Ideas

- Represent user messages as text features (bag-of-words + TF-IDF)
- Train a **Multinomial Naive Bayes** classifier on a small set of labelled examples
- Use the classifier to predict the intent of new user inputs
- Map intents to responses with a simple rule-based layer

## Project Structure

```text
simple-chatbot-python/
│── src/
│   └── chatbot.py
│── requirements.txt
└── README.md
