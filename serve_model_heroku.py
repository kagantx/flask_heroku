from flask import Flask
from flask import request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load pickle file
with open('model.pkl', 'rb') as file:
    lr = pickle.load(file)


@app.route('/predict_single', methods=['POST'])
def single_pred():
    """Obtain a single prediction from user input in the form of a string
    The form of the string is 010000101111000...
    To get the prediction, we convert it to a numpy array in the fastest way.

    """
    string = request.args.get('picture')
    query = np.fromiter(string, dtype=np.int8).reshape(1, -1)
    prediction = lr.predict(query)
    return prediction[0]


@app.route('/predict_mult', methods=['POST'])
def mult_pred():
    """Obtain multiple predictions using json input. The json input format is just a list of lists
    with each list being 768 items long for each pixels"""
    my_json = request.json
    query = np.array(my_json, dtype=np.int8)
    result = lr.predict(query)
    return jsonify([int(pred) for pred in result])


if __name__ == "__main__":
    port=os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0',port=int(port))
    else:
        app.run()
