import my
import os
import pickle
from flask import Flask, render_template, request
from waitress import serve
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def go_to_main():
    return render_template('main.html')

@app.route('/model')
def create_model():
    pass

print('http://127.0.0.1:5011')
serve(app, host='localhost', port='5011')
