from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import pickle

MAXLEN = 50
THRESHOLD = 0.626

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        data = json_data

        # If get only one string
        if type(data) == str:
            data = [data]

        # Preprocessing data
        data = keras_tokenizer.texts_to_sequences(data)
        data = keras.preprocessing.sequence.pad_sequences(data,
                                                        padding='post',
                                                        maxlen=MAXLEN)

        predict = model.predict(data)
        labels = np.where(np.array(predict) > THRESHOLD, 'About', 'None')

        responces = pd.Series(labels.reshape(-1)).to_json()
        responces = jsonify(responces)

        responces.status_code = 200

        return responces

    except Exception as e:
        raise e





if __name__ == '__main__':

    # Load models
    print('Load models...')
    model = keras.models.load_model('emb_model')
    with open('keras_tokenizer.pkl', 'rb') as file:
        keras_tokenizer = pickle.load(file)

    print('Run server...')
    app.run(debug=True)


