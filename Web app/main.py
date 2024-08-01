from flask import Flask, request, jsonify, send_file
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import pickle

def preprocess_text(X):
    stop = set(stopwords.words('english') + list(string.punctuation))

    if isinstance(X, str):
        X = np.array([X])
        
    X_preprocessed = []

    for i, sms in enumerate(X):
        sms = np.array([i.lower() for i in word_tokenize(sms) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(sms)
        
    if len(X) == 1:
        return X_preprocessed[0]
    
    return X_preprocessed

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

    return p_word_given_class

def prob_sms_given_class(treated_sms, cls, word_frequency, class_frequency):
    prob = 1

    for word in treated_sms:
        if word in word_frequency.keys(): 
            prob *= prob_word_given_class(word, cls, word_frequency, class_frequency)

    return prob

def log_prob_sms_given_class(treated_sms, cls, word_frequency, class_frequency):
    prob = 0

    for word in treated_sms: 
        if word in word_frequency.keys(): 
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))

    return prob

def log_naive_bayes(treated_sms, word_frequency, class_frequency, return_likelihood = False):    
    log_prob_sms_given_spam = log_prob_sms_given_class(treated_sms, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency) 
    log_prob_sms_given_ham = log_prob_sms_given_class(treated_sms, cls = 'ham',word_frequency = word_frequency, class_frequency = class_frequency) 

    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam']) 
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam']) 

    log_spam_likelihood = np.log(p_spam) + log_prob_sms_given_spam 
    log_ham_likelihood = np.log(p_ham) + log_prob_sms_given_ham 

    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)

    if log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0

with open('parameters.pkl', 'rb') as f:
    word_frequency, class_frequency = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/check', methods=['POST'])
def spam_checker():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON provided'}), 400
        
        input_sms = data.get('sms', '')
        
        if not input_sms:
            return jsonify({'error': 'No SMS provided'}), 400
        
        input_sms = preprocess_text(input_sms)
        prediction = log_naive_bayes(input_sms, word_frequency, class_frequency)
        result = "The SMS is likely to be spam" if prediction == 1 else "The SMS is unlikely to be spam"
        return jsonify({'result': result})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
