import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

lemmatizer = WordNetLemmatizer()
words = pickle.load(open('./models/words.pkl', 'rb'))
classes = pickle.load(open('./models/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))
