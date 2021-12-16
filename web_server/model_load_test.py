import my

print('모델 부르는 중')
model_two = my.aws.load_model_from_s3('two/model', 'ava-data-model-main')
print('모델 부르기 끝')