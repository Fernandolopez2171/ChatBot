import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

def load_data(file_path):
    """
    Carga los datos desde el archivo 'intents.json' ubicado en la ruta 'file_path'.   
     
    Parámetros
    ------------
    file_path : str
        Ruta completa donde se encuentra el archivo 'intents.json'. 

    Return
    ------------
    data : dict
        Diccionario con los datos cargados desde el archivo 'intents.json'.
        
    Raises
    ------------
    FileNotFoundError
        Si no se encuentra el archivo 'intents.json' en la ruta 'file_path'.
    """
    try:
        with open(file_path, 'r') as f:
            print(f"Loading data from '{file_path}'")
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"No such file or directory: '{file_path}'")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
def preprocess_data(data):
    """
    Preprocesa los datos de entrada para convertirlos en un DataFrame de pandas.

    Parámetros
    ----------
    data : Diccionario que contiene los datos a preprocesar. Se espera que tenga una clave 'intents' que contenga una lista de diccionarios.
    Cada diccionario debe tener las claves 'patterns', 'responses' y 'tag'.

    Return
    ------
    df : pandas.DataFrame
        DataFrame de pandas que contiene los datos preprocesados. Cada fila corresponde a un patrón y contiene la etiqueta correspondiente ('tag'),
        el patrón ('patterns') y las respuestas ('responses').

    Ejemplo
    -------
    >>> data = {
    ...     "intents": [
    ...         {"tag": "greeting", "patterns": ["Hi", "Hello"], "responses": ["Hello!", "Hi there!"]},
    ...         {"tag": "goodbye", "patterns": ["Bye", "See you"], "responses": ["Goodbye!", "See you later!"]}
    ...     ]
    ... }
    >>> preprocess_data(data)
       tag patterns        responses
    0  greeting       Hi  [Hello!, Hi there!]
    1  greeting    Hello  [Hello!, Hi there!]
    2   goodbye      Bye  [Goodbye!, See you later!]
    3   goodbye  See you  [Goodbye!, See you later!]
    """
    print("Preprocessing data...")
    df = pd.DataFrame(data['intents'])
    dic = {"tag":[], "patterns":[], "responses":[]}
    for i in range(len(df)):
        ptrns = df[df.index == i]['patterns'].values[0]
        rspns = df[df.index == i]['responses'].values[0]
        tag = df[df.index == i]['tag'].values[0]
        for j in range(len(ptrns)):
            dic['tag'].append(tag)
            dic['patterns'].append(ptrns[j])
            dic['responses'].append(rspns)
    df = pd.DataFrame.from_dict(dic)
    df['tag'].unique()
    return df

def process_and_encode(df):
    """
    Procesa y codifica los patrones y las etiquetas en el DataFrame proporcionado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos. Se espera que tenga las columnas 'patterns' y 'tag'.

    Return
    ------
    X : numpy.ndarray
        Secuencias de patrones codificadas y rellenadas.
    y : numpy.ndarray
        Etiquetas codificadas.
    """
    if 'patterns' not in df.columns or 'tag' not in df.columns:
        raise ValueError("DataFrame must contain 'patterns' and 'tag' columns")

    print("Processing and encoding data...")
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(df['patterns'])
    tokenizer.get_config()
    vacab_size = len(tokenizer.word_index)
    #print('number of unique words =', vacab_size)

    ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
    X = pad_sequences(ptrn2seq, padding='post')
    #print('X shape =', X.shape)

    lbl_enc = LabelEncoder()
    y = lbl_enc.fit_transform(df['tag'])
    #print('y shape =', y.shape)
    #print('num of classes =', len(np.unique(y)))

    return X, y, vacab_size , tokenizer, lbl_enc

def loadModel(model_path):
    model = load_model(model_path)
    return model