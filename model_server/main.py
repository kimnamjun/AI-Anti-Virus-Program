import my
import os
from datetime import datetime

os.makedirs('./checkpoint/', exist_ok=True)
os.makedirs('./temp/', exist_ok=True)

print('모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

train_filenames = [f'train_features_{i}.jsonl' for i in range(6)]
test_filenames = ['test_features.jsonl']

my.one.create_model(train_filenames, test_filenames)
my.two.create_model(train_filenames, test_filenames)

print('모델 생성 종료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
