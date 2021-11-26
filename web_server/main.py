import my
import os
from datetime import datetime
from flask import Flask, render_template, request
from waitress import serve
from werkzeug.utils import secure_filename


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # setting
        tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        props = my.aws.load_from_s3('one/props.pickle', 'ava-data-model')
        model = my.aws.load_from_s3('one/model.pickle', 'ava-data-model')

        # file upload
        file = request.files['file']
        filename = secure_filename(file.filename)
        filename_path = os.path.join('./dataset/temp', filename)
        file.save(os.path.join(filename_path))

        # file to json
        json_file = my.file2pe.convert_file_to_pe(filename_path)
        if not json_file:
            return render_template('/error', error_code=1, error_msg='파일 변환 실패')
        temp_file_name1 = './dataset/temp/temp.json'
        with open(temp_file_name1, 'w') as file:
            file.write(json_file)

        # json to df
        df = my.preprocessing_one.convert_json_to_df(temp_file_name1)
        df = my.preprocessing_one.reduce_features(df, props)
        df = df.set_index('sha256')
        df.to_csv('./dataset/temp/pca_df.csv', header=True)  # header=True 맞나여?

        # predict
        x, y = df.drop('label', axis=1), df['label']
        result = model.predict(x)[0]

        # save to aws
        my.aws.save_to_s3('./dataset/temp/temp.json', 'ava-data-json', f'{filename}_{tm}.json')
        my.aws.save_to_dynamo('./dataset/temp/pca_df.csv', 'AVA-01')

    except Exception as err:
        raise err
    finally:
        path = './dataset/temp/'
        for filename in os.listdir(path):
            os.remove(path + filename)

    return render_template('result.html', result=result)


@app.errorhandler(404)
def error(error):
    return render_template('/error.html', error_code=404), 404


print('http://127.0.0.1:5021')
serve(app, host='localhost', port='5021')
