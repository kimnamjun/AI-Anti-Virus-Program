import my
import os
from datetime import datetime
from flask import Flask, render_template, request
from waitress import serve
from werkzeug.utils import secure_filename
import boto3


app = Flask(__name__)
s3c = boto3.client('s3')


@app.route('/')
def go_to_main():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

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
    props = my.aws.load_from_s3('one/props.pickle', 'ava-data-model')
    model = my.aws.load_from_s3('one/model.pickle', 'ava-data-model')
    # with open('./model/properties.pickle', 'rb') as props, open('./model/rf_model.pickle', 'rb') as model:
    #     props = pickle.load(props)
    #     model = pickle.load(model)

    df = my.preprocessing_one.json2df(temp_file_name1)
    df = my.preprocessing_one.reduce_features(df, props)
    df = df.set_index('sha256')

    x, y = df.drop('label', axis=1), df['label']
    result = model.predict(x)[0]

    # data storage
    df.to_csv('./dataset/temp/pca_df.csv', index=True, header=True)

    s3c.upload_file(temp_file_name1, 'ava-data-json', f'{filename}_{tm}.json')

    my.csv2ddb.csv_to_dynamo()


    return render_template('result.html', result=result)


@app.errorhandler(404)
def error(error):
    return render_template('/error.html', error_code=404), 404


print('http://127.0.0.1:5021')
serve(app, host='localhost', port='5021')
