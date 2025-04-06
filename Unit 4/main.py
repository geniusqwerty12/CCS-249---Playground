# generated from ChatGPT 

import preprocess
from naive_bayes_classifier import NaiveBayesClassifier

# Sample dataset
data = [
    ("buy cheap now", "spam"),
    ("limited offer for you", "spam"),
    ("click here to win", "spam"),
    ("meeting scheduled for tomorrow", "ham"),
    ("team meeting in the office", "ham"),
    ("can you send me the files", "ham")
]


# Tokenize dataset
tokenized_data = [(preprocess(text), label) for text, label in data]

# Display tokenized data
print(tokenized_data)

# Train the model
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(tokenized_data)

# Sample test data
test_samples = [
    "cheap offer available now",
    "send me the files please",
    "click here to claim your prize"
]

# Predictions
for sample in test_samples:
    prediction = nb_classifier.predict(sample)
    print(f"'{sample}' → Predicted as: {prediction}")