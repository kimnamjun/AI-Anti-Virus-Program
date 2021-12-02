import pandas as pd
from pycaret.classification import *
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def create_voting_model(train_df, test_df):
    train_df = train_df[train_df['label'] != -1].set_index('sha256')
    test_df = test_df[test_df['label'] != -1].set_index('sha256')

    setup(data=train_df.sample(frac=0.1), target='label', silent=True)
    selected_models = compare_models(n_select=3)
    print('selected models', selected_models, sep='\n')

    params = list()
    for model in selected_models:
        parameters = dict()
        for key, value in model.get_params().items():
            if value is None:
                continue
            parameters[key] = [value]
        params.append(parameters)

    cv = StratifiedKFold(shuffle=True)
    models = [GridSearchCV(estimator=selected_models[i], param_grid=params[i], cv=cv) for i in range(len(selected_models))]

    voting_model = VotingClassifier(estimators=[[str(model.estimator).split('(')[0], model] for model in models])

    x_train, y_train = train_df.drop('label', axis=1), train_df['label']
    x_test, y_test = test_df.drop('label', axis=1), test_df['label']

    voting_model.fit(x_train, y_train)

    y_pred = voting_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('voting model accuracy :', acc)

    return voting_model


def create_tfidf_and_logistic_regression_model(train_df, test_df):
    train_df['corpus'] = [' '.join(imports) for imports in train_df['imports']]
    test_df['corpus'] = [' '.join(imports) for imports in test_df['imports']]

    tfidf_vectorizer = TfidfVectorizer()
    x_train = tfidf_vectorizer.fit_transform(train_df['corpus'])
    x_test = tfidf_vectorizer.transform(test_df['corpus'])
    y_train = train_df['label']
    y_test = test_df['label']

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print('SECOND model accuracy :', acc)

    return tfidf_vectorizer, model
