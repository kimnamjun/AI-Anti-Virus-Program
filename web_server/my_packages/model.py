def predict_one(df, model):
    x = df
    y_pred = model.predict(x)
    return y_pred


def predict_two(df, vectorizer, model):
    df['corpus'] = [' '.join(imports) for imports in df['imports']]
    x = vectorizer.transform(df['corpus'])
    y_pred = model.predict(x)
    return y_pred
