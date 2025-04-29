from pos_hmm import HMM

# Training data (very small for illustration)
tagged_data = [
    [('I', 'PRON'), ('run', 'VERB')],
    [('She', 'PRON'), ('eats', 'VERB')],
    [('The', 'DET'), ('cat', 'NOUN')],
    [('A', 'DET'), ('dog', 'NOUN')]
]

model = HMM()
model.train(tagged_data)

# Prediction
sentence = ['I', 'eat']
tags = model.viterbi(sentence)
print("Sentence:", sentence)
print("Predicted Tags:", tags)