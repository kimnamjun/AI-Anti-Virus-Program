import my
import os
from datetime import datetime
from flask import Flask, redirect, render_template, request, url_for
from waitress import serve
from werkzeug.utils import secure_filename

app = Flask(__name__)
os.makedirs('./temp/', exist_ok=True)
os.makedirs('./model/', exist_ok=True)

props_one = my.aws.load_pickle_from_s3('one/properties.pickle', 'ava-data-model-main')
model_one = my.aws.load_pickle_from_s3('one/voting_model.pickle', 'ava-data-model-main')
props_two = my.aws.load_pickle_from_s3('two/properties.pickle', 'ava-data-model-main')
model_two = my.aws.load_model_from_s3()


@app.route('/')
def index():
    return render_template('/main.html')


@app.route('/home')
def goto_home():
    return redirect(url_for('index'))


@app.route('/demonstration')
def goto_demonstration():
    return render_template('/AVA/home/home-default.html')


@app.route('/profiles')
def goto_profiles():
    return render_template('/AVA/pages/page-profile-users-2.html')


@app.route('/FAQ')
def goto_FAQ():
    return render_template('/AVA/pages/page-faq-2.html')


@app.route('/contacts')
def goto_contacts():
    return render_template('/AVA/pages/page-contacts-2.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # file upload
        file = request.files['file']
        filename = secure_filename(file.filename)
        filename_path = './temp/' + filename
        file.save(filename_path)

        # file to json
        json_file = my.file2pe.convert_file_to_pe(filename_path)
        if not json_file:
            return render_template('/error', error_code=1, error_msg='파일 변환 실패')
        json_filename = f'_{filename}_{tm}'.replace('.', '_') + '.json'
        with open(f'./temp/{json_filename}', 'w') as file:
            file.write(json_file)
        my.aws.save_to_s3(f'./temp/{json_filename}', 'ava-data-json-main', json_filename)

        df1, df2 = my.preprocess.convert_json_to_df(json_filename)

        df1 = my.preprocess.reduce_features(df1, props_one)
        x1 = df1.drop(['sha256', 'label'], axis=1)
        result1 = my.model.predict_one(x1, model_one)[0]

        df2 = my.preprocess.preprocess_api(df2, props_two)
        result2 = my.model.predict_two(df2, model_two)[0][0]

        my.aws.save_to_dynamo(df1, 'AVA-01')
        my.aws.save_to_dynamo(df2, 'AVA-02')

        result = '정상 파일' if ((result1 + result2) / 2) < 0.5 else '악성 파일'

    except Exception as err:
        raise err
    finally:
        path = './temp/'
        for filename in os.listdir(path):
            os.remove(path + filename)

    return redirect(url_for('goto_result', result=result), code=307)


@app.route('/result', methods=['POST'])
def goto_result():
    result = request.args['result']
    return render_template('result.html', result=result)


@app.errorhandler(404)
def error(error):
    return render_template('/error.html', error_code=404, error_msg='찾을 수 없는 페이지'), 404


@app.errorhandler(405)
def error(error):
    return render_template('/error.html', error_code=405, error_msg='허용되지 않은 요청'), 405


@app.errorhandler(500)
def error(error):
    return render_template('/error.html', error_code=500, error_msg='서버 문제'), 500


print('Flask app is running')
serve(app, host='0.0.0.0', port='5021')
