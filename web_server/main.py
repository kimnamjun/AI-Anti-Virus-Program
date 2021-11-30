import my
import os
import pickle
from datetime import datetime
from flask import Flask, redirect, render_template, request, url_for
from waitress import serve
from werkzeug.utils import secure_filename


app = Flask(__name__)
props_one = my.aws.load_from_s3('one/properties.pickle', 'ava-data-model')
model_one = my.aws.load_from_s3('one/model.pickle', 'ava-data-model')
props_two = my.aws.load_from_s3('two/properties.pickle', 'ava-data-model')
vectorizer_two = my.aws.load_from_s3('two/vectorizer.pickle', 'ava-data-model')
model_two = my.aws.load_from_s3('two/model.pickle', 'ava-data-model')
print('Loaded successfully from s3')


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # file upload
        file = request.files['file']
        filename = secure_filename(file.filename)
        filename_path = os.path.join('./dataset/temp', filename)
        file.save(os.path.join(filename_path))

        # file to json
        json_file = my.file2pe.convert_file_to_pe(filename_path)
        if not json_file:
            return render_template('/error', error_code=1, error_msg='파일 변환 실패')
        json_file_name = './dataset/temp/temp.json'
        with open(json_file_name, 'w') as file:
            file.write(json_file)

        df1 = my.preprocessing_one.convert_json_to_df(json_file_name)
        df1 = my.preprocessing_one.reduce_features(df1, props_one)
        df1.to_csv('./dataset/temp/df_one.csv', index=False)

        x1 = df1.drop(['sha256', 'label'], axis=1)
        result1 = my.model.predict_one(x1, model_one)

        df2 = my.preprocessing_two.preprocess(json_file_name, props_two)
        with open('./dataset/temp/df_two.pickle', 'wb') as file:
            pickle.dump(df2, file)

        x2 = df2.drop(['sha256', 'label'], axis=1)
        result2 = my.model.predict_two(x2, vectorizer_two, model_two)

        # save to aws
        my.aws.save_to_s3('./dataset/temp/temp.json', 'ava-data-json', f'{filename}_{tm}.json')
        my.aws.save_to_dynamo('./dataset/temp/df_one.csv', 'AVA-01')
        # my.aws.save_to_dynamo('./dataset/temp/df_two.csv', 'AVA-01')

    except Exception as err:
        raise err
    finally:
        path = './dataset/temp/'
        for filename in os.listdir(path):
            os.remove(path + filename)

    return redirect(url_for('result.html', result1=result1, result2=result2), code=307)


@app.route('/result', methods=['POST'])
def result():
    result1 = request.form['result1']
    result2 = request.form['result2']
    return render_template('result.html', result1=result1, result2=result2)


@app.errorhandler(404)
def error(error):
    return render_template('/error.html', error_code=404, error_msg='찾을 수 없는 페이지'), 404


@app.errorhandler(405)
def error(error):
    return render_template('/error.html', error_code=405, error_msg='허용되지 않은 요청'), 405


print('Flask app is running')
serve(app, host='0.0.0.0', port='5021')
