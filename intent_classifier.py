# intent_classifier.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Training data
training_sentences = [
    "I want to book a cleaning",
    "Can I get a teeth whitening appointment",
    "Please cancel my booking",
    "I need to reschedule my appointment",
    "Book me for a checkup",
    "I want to cancel my appointment",
    "I need to change the time of my appointment",
    "Schedule a dental cleaning",
    "I'd like to whiten my teeth",
    "Move my appointment to another time"
]

training_labels = [
    "book_cleaning",
    "book_whitening",
    "cancel_appointment",
    "reschedule_appointment",
    "book_checkup",
    "cancel_appointment",
    "reschedule_appointment",
    "book_cleaning",
    "book_whitening",
    "reschedule_appointment"
]

# 2. Convert text to numbers
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# 3. Train the model
model = MultinomialNB()
model.fit(X_train, training_labels)

# 4. Predict function
def predict_intent(user_input):
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]
    confidence = model.predict_proba(X_test).max()
    return prediction, confidence

# 5. Command line test
if __name__ == "__main__":
    print("Type a message to test intent detection ('exit' to quit):\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        intent, confidence = predict_intent(user_input)
        print(f"🔍 Intent: {intent}  |  Confidence: {confidence:.2f}\n")
