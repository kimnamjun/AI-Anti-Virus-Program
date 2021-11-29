import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def create_random_forest_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> RandomForestClassifier:
    """
    Random Forest 모델을 생성합니다.
    :param train_df: 학습용 데이터입니다.
    :param test_df: 학습용 데이터입니다.
    :return: 생성된 Random Forest 모델입니다.
    """
    train_df.set_index('sha256', inplace=True)
    test_df.set_index('sha256', inplace=True)
    x_train, y_train = train_df.drop('label', axis=1), train_df['label']
    x_test, y_test = test_df.drop('label', axis=1), test_df['label']

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print('accuracy :', acc)

    return model
