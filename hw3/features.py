from sklearn.feature_extraction import DictVectorizer


def create_X(df):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    return X, dv