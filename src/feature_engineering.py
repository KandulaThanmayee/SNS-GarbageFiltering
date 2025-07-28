from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def fit_transform_tfidf(texts, save_path=None):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(texts)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
    return X_vect, vectorizer
