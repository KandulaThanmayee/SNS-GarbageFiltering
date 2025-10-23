# 🗑️ Garbage Data Filtering

A Python-based machine learning application that filters irrelevant or "garbage" data such as spam, ads, or low-value posts from social media content using multiple classifiers and a GUI interface. The goal is to improve downstream data quality for analysis and natural language processing (NLP) tasks.

---

## 🚀 Features

- Train and compare multiple classifiers:
  - ✅ Random Forest
  - ✅ Decision Tree
  - ✅ Naive Bayes (Spark)
  - ✅ XGBoost
- TF-IDF vectorization of input text
- GUI-powered interaction via Tkinter
- Word preprocessing using NLTK (stopwords, lemmatization)
- Model evaluation and accuracy comparison
- Easy testing on bulk data (e.g., tweets or posts)

---

## 🛠️ Installation and Setup

Follow these steps to install and run the project successfully.

### 1️⃣ Clone the Repository
cd garbage-data-filtering

### 2️⃣ Set Up the Environment

#### Option A: Using Conda (Recommended)

conda env create -f AOA.yaml
conda activate AOA


#### Option B: Using Pip

pip install -r requirements.txt


### 3️⃣ Download NLTK Resources

After installing dependencies, start a Python shell or notebook and run:

import nltk nltk.download(‘stopwords’) nltk.download(‘wordnet’)


### 4️⃣ Prepare Required Files

Ensure the following files exist:

/data/dataset.csv        # Labeled dataset for training /data/TestData.csv       # Raw/unlabeled test data /models/                 # Models and TF-IDF will be saved here after training


You can manually create the `/models` folder; it will also be auto-populated when models are trained.

---

## 📦 Running the Application

To launch the application:

python main.py


This will open a GUI window. You can now:
- Train your models
- Evaluate model accuracy
- Load test data
- Submit test predictions

---

## 📊 Dataset Format

Your primary dataset **`dataset.csv`** should follow this format:

tweet,label “This is a spam offer!”,0 “Checkout our new product”,1 “Environment is changing drastically”,2


Label classification:
- `0` — Garbage / Irrelevant
- `1` — Advertisement / Promotional
- `2` — Definite / Relevant

Your test dataset (`TestData.csv`) should only contain tweets (no labels or headers if testing inference).

---

## 🖼️ Screenshots

Here's what the GUI looks like:

![GUI Preview](screenshots/gui.png) <!-- Upload actual screenshot image to this path -->

---

## 📁 Project Structure

GarbageDataFiltering
├── main.py
├── config.py
├── AOA.yaml
├── requirements.txt
├── /data/
│   ├── dataset.csv
│   └── TestData.csv
├── /models/
│   └── (trained models + vectorizer saved here)
├── /src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluation.py
├── /gui/
│   └── gui_app.py
└── /utils/
└── logger.py


---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork this repo
2. Create a new branch (`git checkout -b my-feature`)
3. Make your changes
4. Commit and push (`git commit -am 'Add feature' && git push`)
5. Open a pull request 🚀

Please see the `CONTRIBUTING.md` (optional) for guidelines.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 🙋 Support

If you have questions or suggestions, feel free to open an issue or submit a pull request.  
Let’s clean up garbage data together! 💬

---
