import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def create_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: str) -> RandomForestClassifier:
    train_df = train_df[train_df['label'].isin([0, 1])]
    test_df = test_df[test_df['label'].isin([0, 1])]

    x_train, y_train = train_df.drop('label', axis=1), train_df['label']
    x_test, y_test = test_df.drop('label', axis=1), test_df['label']

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print('Random Forest')
    print('Accuracy Score :', accuracy_score(y_test, y_pred))
    print('F1 Score       :', f1_score(y_test, y_pred))

    with open(save_path, 'wb') as file:
        pickle.dump(model, file)
        print(save_path, '에 model 파일이 저장되었습니다.')

    return model


def predict_with_random_forest(test_df: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    x_test, y_test = test_df.drop('label', axis=1), test_df['label']
    y_pred = model.predict(x_test)
    result = pd.DataFrame([y_test.tolist(), y_pred.tolist()], index=['True', 'Pred'], columns=y_test.index).T
    return result
