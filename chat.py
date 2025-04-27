import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import re
import random
from keras.preprocessing.sequence import pad_sequences

#personal imports
from Utils.data_utils import load_data, preprocess_data, process_and_encode, loadModel

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
    
    print("tag: {}".format(tag))
    print("model: {}".format(random.choice(responses)))
        
def main():
    model_path = './Model/chatbot_model.keras'
    model = loadModel(model_path)
    if model is None:
        print("Model not found")
        return
    else:
        print("Model loaded successfully")
        
    data_path = './DataSet/intents.json'
    df = load_data(data_path)
    df = preprocess_data(df)
    
    X, y, vacab_size,tokenizer,lbl_enc = process_and_encode(df)
    
    while True:
        print("--------------------------------------")
        user_input = input("You: ")
        
        if user_input == '-1':
            break
        else:
            generate_answer(user_input, tokenizer, model, lbl_enc, df, X.shape[1])
 
    
if __name__ == "__main__":
    main()