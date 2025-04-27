import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import re
import random

from flask_cors import CORS
from flask import Flask, request, jsonify
from keras.preprocessing.sequence import pad_sequences # type: ignore

#personal imports
from Utils.data_utils import load_data, preprocess_data, process_and_encode, loadModel

app = Flask(__name__)
CORS(app)

#global variables
global model, tokenizer, lbl_enc, max_length, df, X

def load_resources():
    global model, tokenizer, lbl_enc, max_length, df, X
    model_path = './Model/chatbot_model.keras'
    model = loadModel(model_path)
    
    if model is None:
        return False
    
    data_path = './DataSet/intents.json'
    df = load_data(data_path)
    df = preprocess_data(df)
    
    X, _, vocab_size, tokenizer, lbl_enc = process_and_encode(df)
    max_length = X.shape[1]
    return True

def generate_answer(pattern, tokenizer, model, lbl_enc, df, X_shape):
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
        
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    if np.ndim(x_test) == 0:  # if x_test is a 0-d array (scalar)
        x_test = np.expand_dims(x_test, 0)  # make it a 1-d array
    x_test = pad_sequences([x_test], padding='post', maxlen=X_shape)
    y_pred = model.predict(x_test, verbose=0)  # Set verbose to 0
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return tag, random.choice(responses)
    
    
    
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    tag, modelResponse = generate_answer(user_input, tokenizer, model, lbl_enc, df, max_length)
    
    #tag y model son valores nulos error 400
    if tag is None or modelResponse is None:
        response = {
            "tag": tag,
            "modelResponse": modelResponse,
            "Status": "error"
        }
        return jsonify(response), 400
    
    response = {
        "tag": tag,
        "modelResponse": modelResponse,
        "Status": "success"
    }
    
    return jsonify(response)

if __name__ == '__main__':
    if load_resources():
        app.run(debug=True)
    else:
        print("Failed to load resources. Exiting.")