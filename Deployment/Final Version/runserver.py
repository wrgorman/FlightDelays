"""
This script runs the FlaskWebProject1 application using a development server.
"""

from os import environ
from FlaskWebProject1 import app

#import os
import pickle
import pandas as pd
from flask import Flask, request, json, Response

application = Flask(__name__)

filename = 'C:\\temp\\FlightDelayLogRegModel.pik'
infile = open(filename, 'rb')
clf = pickle.load(infile)

@application.route("/")
def home():
    return "Flight Prediction Home Page"

@application.route('/predict', methods=['POST'])
def predict():
    """
        Generates the prediction for titanic
        :return: json output for prediction
    """
    data_row = pd.io.json.json_normalize(request.json)
    prediction = clf.predict(data_row)
    probability = clf.predict_proba(data_row)
    data = {'predict': str(prediction[0]), 'survival_probs': str(probability[0])}
    #data = {'count': str(2)}
    js = json.dumps(data)
    return Response(js, status=200, mimetype='application/json')

@application.route('/health', methods=['GET'])
def health():
    """
        Returns healthy status
        :return: Health status
    """
    js = json.dumps({'status': 'healthy'})
    resp = Response(js, status=200, mimetype='application/json')
    return resp


if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    application.run(HOST, PORT)
