from flask import Flask, render_template, request, redirect
from waitress import serve

app = Flask(__name__)

@app.route('/')
def go_to_index():
    return render_template('index.html')

@app.route('/result')
def go_to_result():
    return render_template('result.html')

print('http://127.0.0.1:5021')
serve(app, host='localhost', port=5021)
