def predict_one(df, model):
    x = df
    y_pred = model.predict(x)
    return y_pred


def predict_two(df, model):
    x = df['imports'].apply(lambda row: ' '.join(row)).to_list()
    y_pred = model.predict(x)
    return y_pred
