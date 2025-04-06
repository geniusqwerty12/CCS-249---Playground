# generated from ChatGPT 

import re

def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()  # Tokenize
