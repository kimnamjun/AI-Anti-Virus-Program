import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def create_random_forest_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> RandomForestClassifier:
    """
    Random Forest 모델을 생성합니다.
    :param train_df: train 데이터입니다. (pca_df)
    :param test_df: test 데이터입니다. (pca_df)
    :return: 생성된 Random Forest 모델입니다.
    """
    x_train, y_train = train_df.drop('label', axis=1), train_df['label']
    x_test, y_test = test_df.drop('label', axis=1), test_df['label']

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print('Random Forest')
    print('Accuracy Score :', accuracy_score(y_test, y_pred))
    print('F1 Score       :', f1_score(y_test, y_pred))

    return model
