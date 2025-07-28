from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(X, y, save_path=None):
    clf = RandomForestClassifier()
    clf.fit(X, y)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
    return clf
