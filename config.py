import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

DATASET_PATH = os.path.join(DATA_DIR, 'dataset.csv')
TESTDATA_PATH = os.path.join(DATA_DIR, 'TestData.csv')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')
