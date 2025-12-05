from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


@dataclass
class IntentExample:
    text: str
    intent: str


def build_training_data() -> List[IntentExample]:
    """
    Define a small set of (text, intent) examples for training the chatbot.
    In a real project this could be loaded from a JSON or CSV file.
    """
    data = [
        # greetings
        IntentExample("hi", "greeting"),
        IntentExample("hello", "greeting"),
        IntentExample("hey there", "greeting"),
        IntentExample("good morning", "greeting"),
        IntentExample("good evening", "greeting"),

        # goodbye
        IntentExample("bye", "goodbye"),
        IntentExample("see you later", "goodbye"),
        IntentExample("good night", "goodbye"),
        IntentExample("talk to you later", "goodbye"),

        # thanks
        IntentExample("thanks", "thanks"),
        IntentExample("thank you", "thanks"),
        IntentExample("that's helpful", "thanks"),
        IntentExample("i appreciate it", "thanks"),

        # name
        IntentExample("what is your name", "name"),
        IntentExample("who are you", "name"),
        IntentExample("tell me your name", "name"),

        # weather (very simple intent, no real API)
        IntentExample("how is the weather", "weather"),
        IntentExample("what's the weather like", "weather"),
        IntentExample("is it raining", "weather"),
        IntentExample("is it sunny today", "weather"),

        # small talk
        IntentExample("how are you", "smalltalk"),
        IntentExample("how is it going", "smalltalk"),
        IntentExample("what's up", "smalltalk"),
    ]
    return data


def prepare_xy(data: List[IntentExample]) -> Tuple[List[str], List[str]]:
    X = [ex.text for ex in data]
    y = [ex.intent for ex in data]
    return X, y


def build_intent_classifier() -> Pipeline:
    """
    Build a simple NLP pipeline:
    - Convert text to token counts
    - Convert counts to TF-IDF features
    - Train a Naive Bayes classifier
    """
    pipeline = Pipeline(
        steps=[
            ("vect", CountVectorizer(lowercase=True)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )
    return pipeline


def train_model() -> Tuple[Pipeline, List[str]]:
    training_data = build_training_data()
    X, y = prepare_xy(training_data)

    model = build_intent_classifier()
    model.fit(X, y)

    intents = sorted(set(y))
    return model, intents


def get_response(intent: str, confidence: float) -> str:
    """
    Map predicted intents to simple responses.
    A real chatbot would handle context and more complex templates.
    """
    responses: Dict[str, List[str]] = {
        "greeting": [
            "Hello! How can I help you today?",
            "Hi there! What would you like to talk about?",
        ],
        "goodbye": [
            "Goodbye! Have a great day.",
            "See you later. 👋",
        ],
        "thanks": [
            "You're welcome!",
            "Happy to help!",
        ],
        "name": [
            "I'm a simple Python chatbot.",
            "You can call me PyBot 🤖.",
        ],
        "weather": [
            "I can't check real-time weather yet, but I hope it's nice where you are!",
            "I'm not connected to a weather API, but it feels like a good day to code.",
        ],
        "smalltalk": [
            "I'm doing great, thanks for asking! What about you?",
            "All good here, just processing some data. 🙂",
        ],
        "fallback": [
            "I'm not sure I understand. Could you rephrase that?",
            "That's interesting! I'm still learning, could you ask in a simpler way?",
        ],
    }

    # If confidence is very low, use fallback
    if confidence < 0.3:
        intent = "fallback"

    candidates = responses.get(intent, responses["fallback"])
    # Pick a deterministic response based on confidence to keep it simple
    index = 0 if confidence < 0.65 else 1
    index = index % len(candidates)
    return candidates[index]


def predict_intent(model: Pipeline, text: str) -> Tuple[str, float]:
    """
    Predict the intent for a given user message and return
    (predicted_intent, confidence_score).
    """
    # Probabilities over classes
    proba = model.predict_proba([text])[0]  # shape: (n_classes,)
    classes = model.classes_

    best_idx = int(np.argmax(proba))
    predicted_intent = classes[best_idx]
    confidence = float(proba[best_idx])

    return predicted_intent, confidence


def chat_loop() -> None:
    """
    Simple console-based chat loop.
    Type 'quit' or 'exit' to stop.
    """
    print("Training intent classifier...")
    model, intents = train_model()
    print(f"Trained on intents: {intents}")
    print("Chatbot is ready! Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Chatbot: Goodbye! 👋")
            break

        if not user_input:
            print("Chatbot: Please type something so I can respond.")
            continue

        intent, confidence = predict_intent(model, user_input)
        response = get_response(intent, confidence)

        print(f"(debug) predicted intent='{intent}', confidence={confidence:.2f}")
        print(f"Chatbot: {response}\n")


def main():
    chat_loop()


if __name__ == "__main__":
    main()
