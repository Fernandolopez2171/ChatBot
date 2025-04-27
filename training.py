import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l2
#personal imports
from Utils.data_utils import load_data, preprocess_data, process_and_encode

logging.basicConfig(level=logging.INFO)

def create_model(input_shape, vocab_size, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Embedding(input_dim=vocab_size+1, output_dim=100, mask_zero=True))
    model.add(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(LayerNormalization())
    model.add(LSTM(32, kernel_regularizer=l2(0.01)))
    model.add(LayerNormalization())
    model.add(Dense(121, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.45))
    model.add(Dense(num_classes, activation="softmax", kernel_regularizer=l2(0.01)))
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def train_model(model, X, y, batch_size, epochs, patience):
    callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patience)]
    model.fit(x=X, y=y, batch_size=batch_size, callbacks=callbacks, epochs=epochs)
    
    return model

def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved at {model_path}")
    
def main():

    try:
        os.makedirs("./DataSet", 0o777, exist_ok=True)
        data = load_data("./DataSet/intents.json")
        if data is not None:
           
            df = preprocess_data(data)
            X, y, vacab_size,_,_ = process_and_encode(df)
            model = create_model(X.shape[1], vacab_size, len(np.unique(y)))
            model = train_model(model, X, y ,batch_size=10, epochs=50, patience=3)

            save_model(model, "./Model/chatbot_model.keras")
            
            
    except Exception as e:
        logging.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()