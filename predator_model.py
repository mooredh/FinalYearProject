from dateutil.parser import parse
import numpy as np
import uuid
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from flask import abort, make_response, jsonify

nltk.download('stopwords')
nltk.download('punkt')

bow = joblib.load('models/bag_of_words_sex_pred.joblib')
vic_pred_model = joblib.load('models/victim_from_predator_model.joblib')
con_based_model = joblib.load('models/conversation_based_model.joblib')

stop_words = set(stopwords.words('english'))
stop_words.add('apos')
stop_words.add('amp')

class PredatorModel:
    def __init__(self, data):
        self.data = data
        self.idx_to_key = list()
        self.idx_to_key_vic = list()

    def get_conversation_based(self):
        x = []
        for conversation_id, conversation in self.data.items():
            x1 = ""
            for text_line in conversation:
                if text_line['text']:
                    x1 += " " + text_line['text']
            self.idx_to_key.append(conversation_id)
            x.append(x1)
        return bow.transform(x).toarray()

    def get_victim_from_predator(self, only_pred_data):
        x = []

        for conversation_id, conversation in only_pred_data.items():
            x1 = dict()
            x11 = []
            x11_key = []
            for text_line in conversation:
                if text_line['text']:
                    if text_line['author'] in x1:
                        x1[text_line['author']] += " " + text_line['text']
                    else:
                        x1[text_line['author']] = text_line['text']
            if len(x1) < 2:
                continue
            for key, val in x1.items():
                x11_key.append(key)
                x11.append(val)
            self.idx_to_key_vic.append(x11_key)
            x.append(x11)

        for i in range(len(x)):
            x[i] = bow.transform(x[i]).toarray().flatten()
        return np.stack(x)

    def count_authors(self, chat):
        authors = set()
        for text_chat in chat:
            authors.add(text_chat['author'])

        return len(authors)

    def remove_stop_words(self):
        for conversation_id, conversation in self.data.items():
            for text_line in conversation:
                if not text_line['text']:
                    text_line['text'] = ""
                word_tokens = word_tokenize(text_line['text'])
                filtered_sentence = []

                for w in word_tokens:
                    if w.lower() not in stop_words and (w.isalnum() or w == '#' or w == '+' or w == '_'):
                        filtered_sentence.append(w.lower())

                text_line['text'] = ' '.join(filtered_sentence)

    def clean(self):
        for key, val in self.data.items():
            count = self.count_authors(val)
            if count != 2:
                response = make_response(jsonify(message="All conversations can have only two users"), 400)
                abort(response)

        self.remove_stop_words()

    def predict(self):
        con_pred = con_based_model.predict(self.get_conversation_based())
        result = dict()
        only_preds = dict()

        for i in range(len(con_pred)):
            con_id = self.idx_to_key[i]
            result[con_id] = dict()
            result[con_id]["predator_detected"] = int(con_pred[i])
            if con_pred[i] == 1:
                only_preds[con_id] = self.data[con_id]

        if len(only_preds) > 0:
            vic_pred = vic_pred_model.predict(self.get_victim_from_predator(only_preds))

            for i in range(len(vic_pred)):
                con_id = self.idx_to_key_vic[i]
                result[con_id]["predator"] = vic_pred[i]

        return result

