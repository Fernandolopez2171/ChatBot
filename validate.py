import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report

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
    if np.ndim(x_test) == 0:
        x_test = np.expand_dims(x_test, 0)
    x_test = pad_sequences([x_test], padding='post', maxlen=X_shape)
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
  
    return tag , pattern

def main():
    model_path = './Model/chatbot_model2.keras'
    model = loadModel(model_path)
    if model is None:
        print("Model not found")
        return
    else:
        print("Model loaded successfully")
        
    data_path = './DataSet/intents.json'
    validate_data_path = "./DataSet/validate.json"
    df = load_data(data_path)
    df = preprocess_data(df)
    X, y, vacab_size,tokenizer,lbl_enc = process_and_encode(df)
    
    df_validate = load_data(validate_data_path)
    df_validate = preprocess_data(df_validate)
    true_tags = []
    predicted_tags = []
    
    for i in range(len(df_validate)):
        tag, pattern = generate_answer(df_validate['patterns'][i], tokenizer, model, lbl_enc, df, X.shape[1])
        true_tags.append(df_validate['tag'][i])
        predicted_tags.append(tag)
 

    unique_labels = sorted(set(true_tags + predicted_tags))


    conf_matrix = confusion_matrix(true_tags, predicted_tags, labels=unique_labels)

    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    report = classification_report(true_tags, predicted_tags, labels=unique_labels, zero_division=1, output_dict=True)


    report.pop('macro avg', None)
    report.pop('weighted avg', None)
    report.pop('accuracy', None)


    df_report = pd.DataFrame(report).transpose()

    print("Classification Report:")
    print(df_report)
if __name__ == "__main__":
    main()