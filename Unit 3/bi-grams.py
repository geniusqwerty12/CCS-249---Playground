# Code generate by ChatGPT
from nltk import bigrams
from nltk.tokenize import word_tokenize

from collections import Counter

# Extracting bi-grams from the text
text = "I love machine learning and artificial intelligence."
tokens = word_tokenize(text)  # Tokenizing words
print(tokens)

# Generating b-gram tokens
bigram_list = list(bigrams(tokens))  # Generating bigrams
print(bigram_list)

# Generating the Bigram Model
# Calculate the probabilties
def bigram_probabilities(text):
    tokens = word_tokenize(text.lower()) 
    bigram_counts = Counter(bigrams(tokens))
    unigram_counts = Counter(tokens)

    # compute the probabilities
    # create a map variable to store the probabilities
    bigram_probs = {bigram: count / unigram_counts[bigram[0]] 
                    for bigram, count in bigram_counts.items()}
    
    return bigram_probs

text = "the dog barks. the dog runs. the cat meows."

bigram_probs = bigram_probabilities(text)

for bigram, prob in bigram_probs.items():
    print(f"P({bigram[1]} | {bigram[0]}) = {prob: .4f}")


# Testing the model
# 1) Text Prediction
# param 1: bigram model
# param 2: last word on the sentence
def predict_next_word(bigram_probs, current_word):
    # Retrieve the items from the bigram model that matches the first of the bigram with the current word 
    candidates = { k[1]: v for k, v in bigram_probs.items() if k[0] == current_word }
    if not candidates:
        return None # if there are no matches return nothing

    # get the key-value pair with the highest probability
    return max(candidates, key=candidates.get)

# Predict the next word
predicted_word = predict_next_word(bigram_probs, "barks")
print(f"Predicted next word after the word : {predicted_word}")

# Text Generation
def generate_bigram_text(bigram_model, start_word, length=10):
    current_word = start_word.lower()
    generated_text = [current_word]

    for _ in range(length):
        # fetch the probability in the model that contains the start word
        candidates = { k[1]: v for k, v in bigram_model.items() if k[0] == current_word }
        # get the one with the highest probability
        current_word = max(candidates, key=candidates.get)
        generated_text.append(current_word)
    
    # combine the array to a string values
    return " ".join(generated_text)

print(f"Generated Text with the length of 10 is: ", generate_bigram_text(bigram_probs,"the", 10))