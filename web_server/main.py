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

@app.route('/test')
def go_to_test():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    # file upload
    file = request.files['file']
    filename = secure_filename(file.filename)
    filename_path = os.path.join('./dataset/temp', filename)
    file.save(os.path.join(filename_path))

    # file to json
    json_file = my.file2pe.file2pe(filename_path)
    if not json_file:
        return render_template('/error', error_code=1, error_msg='파일 변환 실패')
    temp_file_name1 = './dataset/temp/temp.json'
    with open(temp_file_name1, 'w') as file:
        file.write(json_file)

    # predict one
    with open('./model/properties.pickle', 'rb') as file:
        props = pickle.load(file)
    with open('./model/rf_model.pickle', 'rb') as file:
        model = pickle.load(file)

    df = my.preprocessing_one.json2df(temp_file_name1)
    df = my.preprocessing_one.reduce_features(df, props)
    df = df.set_index('sha256')
    x, y = df.drop('label', axis=1), df['label']
    result = model.predict(x)[0]

    return render_template('result.html', result=result)


@app.errorhandler(404)
def error(error):
    return render_template('/error.html', error_code=404), 404

print('http://127.0.0.1:5021/test')
serve(app, host='localhost', port='5021')
