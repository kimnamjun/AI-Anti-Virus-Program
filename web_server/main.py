import time

import my

import os
import pickle
from flask import Flask, render_template, request
from waitress import serve
from werkzeug.utils import secure_filename
import boto3

app = Flask(__name__)

@app.route('/')
def go_to_main():
    return render_template('main.html')


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

    # data storage
    df.to_csv('./dataset/temp/pca_df.csv', index=True, header=True)

    s3 = boto3.client('s3')
    tm = time.localtime(time.time())
    # AWS s3에 파일 업로드
    # 첫번째 매개 변수 : 로컬에서 올릴 파일이름 file.filename (업로드한 파일의 원래 이름)
    # 두번째 매개 변수 : s3 버킷 이름 ( 본인의 버켓 이름을 입력할 것)
    # 세번째 매개 변수 : 버킷에 저장될 파일 이름. ( 업로드한 파일의 원래 이름)
    s3.upload_file("./dataset/temp/temp.json", 'ava-data-json', '파일명' + time.strftime('%Y-%m-%d_%I-%M-%S', tm) + '.json')

    my.csv2ddb.csv_to_dynamo()

    x, y = df.drop('label', axis=1), df['label']
    result = model.predict(x)[0]

    return render_template('result.html', result=result)


@app.errorhandler(404)
def error(error):
    return render_template('/error.html', error_code=404), 404


print('http://127.0.0.1:5021')
serve(app, host='localhost', port='5021')
