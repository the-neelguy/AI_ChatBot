import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import os

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

data_file = open('./data/intents.json').read()
intents = json.loads(data_file)

# Tokenize and lemmatize the words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Ensure the models directory exists
if not os.path.exists('./models/'):
    os.makedirs('./models/')

# Save words and classes
pickle.dump(words, open('./models/words.pkl', 'wb'))
pickle.dump(classes, open('./models/classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Ensure all input vectors (bags) and output vectors have the same length
train_x = np.array([np.array(bag) for bag, _ in training])
train_y = np.array([np.array(output_row) for _, output_row in training])

# Save training data
np.save('./models/train_x.npy', train_x)
np.save('./models/train_y.npy', train_y)
