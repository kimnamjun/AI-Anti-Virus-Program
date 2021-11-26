import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def create_random_forest_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Random Forest 모델을 생성합니다.
    :param df: 학습용 데이터입니다.
    :return: 생성된 Random Forest 모델입니다.
    """
    df.set_index('sha256', inplace=True)
    x, y = df.drop('label', axis=1), df['label']

    model = RandomForestClassifier()
    model.fit(x, y)

    return model
