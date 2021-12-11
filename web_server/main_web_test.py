from flask import Flask, redirect, render_template, request, url_for
from waitress import serve

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('/main.html')


@app.route('/home')
def goto_home():
    return redirect(url_for('index'))


@app.route('/demonstration')
def goto_demonstration():
    return render_template('/unify-main/home/home-default.html')


@app.route('/profiles')
def goto_profiles():
    return render_template('/unify-main/pages/page-profile-users-2.html')


@app.route('/FAQ')
def goto_FAQ():
    return render_template('/unify-main/pages/page-faq-2.html')


@app.route('/contacts')
def goto_contacts():
    return render_template('/unify-main/pages/page-contacts-2.html')


@app.route('/predict', methods=['POST'])
def predict():
    result = 123456
    return redirect(url_for('result.html', result=result), code=307)


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
print('http://127.0.0.1:5021')
serve(app, host='127.0.0.1', port='5021')
