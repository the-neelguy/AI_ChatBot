import keras
import tensorflow
from tensorflow.keras.models import load_model
import json
import numpy as np
import random
import pickle
from scripts.utils import bow, clean_up_sentence

model = load_model('models/chatbot_model.h5')
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

# response = chatbot_response("Need help, which dress should I wear?")
# print(response)

# response = chatbot_response("Hello, how are you?")
# print(response)

# response = chatbot_response("Can u help me?")
# print(response)

# response = chatbot_response("Thanks a lot")
# print(response)

# response = chatbot_response("See you soon !")
# print(response)
