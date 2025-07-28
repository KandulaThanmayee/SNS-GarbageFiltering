import tkinter as tk
from tkinter import messagebox
import pandas as pd

from src import preprocessing, feature_engineering, model
from config import DATASET_PATH, MODEL_PATH, TFIDF_PATH
from sklearn.model_selection import train_test_split

class GarbageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Garbage Data Filtering")
        self.root.geometry("400x200")

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.test_button = tk.Button(root, text="Test Model", command=self.test_model)
        self.test_button.pack(pady=10)

    def train_model(self):
        df = pd.read_csv(DATASET_PATH)
        df = preprocessing.clean_dataset(df)
        X = df['tweet']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_vect_train, tfidf = feature_engineering.fit_transform_tfidf(X_train, save_path=TFIDF_PATH)
        clf = model.train_model(X_vect_train, y_train, save_path=MODEL_PATH)

        messagebox.showinfo("Training", "Model trained and saved successfully.")

    def test_model(self):
        messagebox.showinfo("Status", "Testing not implemented yet.")
