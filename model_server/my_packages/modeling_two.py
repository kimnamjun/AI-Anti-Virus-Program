import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier

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
