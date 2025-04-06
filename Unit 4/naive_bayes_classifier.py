# generated from ChatGPT 

from collections import Counter, defaultdict
import preprocess
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(Counter)
        self.class_totals = defaultdict(int)
        self.vocab = set()
        self.class_priors = {}
    
    def fit(self, data):
        # Count words in each class
        for text, label in data:
            self.class_word_counts[label].update(text)
            self.class_totals[label] += len(text)
            self.vocab.update(text)
        
        # Compute class priors
        total_docs = len(data)
        self.class_priors = {
            label: count / total_docs for label, count in Counter([label for _, label in data]).items()
        }

    def predict(self, text):
        text = preprocess(text)
        class_scores = {}

        # Compute P(y | X) for each class
        for label in self.class_priors:
            # Start with log P(y)
            log_prob = np.log(self.class_priors[label])
            
            # Add log P(X | y) for each word
            for word in text:
                word_count = self.class_word_counts[label][word] + 1  # Laplace smoothing
                total_words = self.class_totals[label] + len(self.vocab)  # Vocabulary size
                log_prob += np.log(word_count / total_words)
            
            class_scores[label] = log_prob
        
        # Return class with highest probability
        return max(class_scores, key=class_scores.get)