from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
import traceback
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from nltk.stem import PorterStemmer
from sklearn.tree import DecisionTreeClassifier
#loading all SPARK packages
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import os
from pyspark.ml.classification import NaiveBayes #loading Naive Bayes classifier from SPARK package
from pyspark.sql.functions import col
from xgboost import XGBClassifier


main = tkinter.Tk()
main.title("Effective Garbage Data Filtering Algorithm for SNS Big Data Processing by Machine Learning")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global rf, X, Y, word_meme
accuracy = []
global X_train, X_test, y_train, y_test

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1 and len(word) < 20]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")

    dataset = pd.read_csv('Dataset/dataset.csv')
    text.insert(END, str(dataset.head()))
    text.update_idletasks()
    label = dataset.groupby('Label').size()
    label.plot(kind="bar")
    plt.title("SNS Graph 0 (Irrelevant or Garbage), 1 (Advertisement) & 2 (Definite Data)")
    plt.show()

def dataClassifier():
    global dataset, word_meme
    text.delete('1.0', END)
    textdata.clear()
    labels.clear()
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'Tweets')
        label = dataset.get_value(i, 'Label')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(label)
        clean = cleanPost(msg)
        textdata.append(clean)
    word_meme = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    weight = word_meme.fit_transform(textdata).toarray()        
    df = pd.DataFrame(weight, columns = word_meme.get_feature_names())
    df['target'] = labels
    df.to_csv("weight.csv", index=False)
    text.insert(END,"Words Morphological Weights\n\n")
    text.insert(END,str(df))
    
def naiveBayesTraining():
    text.delete('1.0', END)
    accuracy.clear()
    try:
        #create spark object using HDFS hadoop big data processing
        spark = SparkSession.builder.appName("HDFS").getOrCreate()
        sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("HDFS")) #creating spark object and initializing it
        #logs = sparkcont.setLogLevel("ERROR")
        filePath = "C:/GarbageDataFiltering/weight.csv"
        print(filePath)
        #read dataset weight file
        df = spark.read.option("header","true").csv("file:///"+filePath,inferSchema=True)
        temp = df.toPandas()
        #extract columns from dataset
        required_features = df.columns
        #convert dataset into spark compatibe format
        assembler = VectorAssembler(inputCols=required_features, outputCol='features',handleInvalid="skip")
        transformed_data = assembler.transform(df)
        indexer = StringIndexer(inputCol="target",outputCol="indexlabel",handleInvalid="skip")
        transformed_data = indexer.fit(transformed_data).transform(transformed_data)
        #split dataset ino train and test where 0.8 refers to 80% training data 0.2 means 20 testing data
        (training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])
        #training spark naive bayes
        nb = NaiveBayes(modelType="multinomial", featuresCol = 'features', labelCol = 'indexlabel')
        nb_model = nb.fit(training_data)
        #predicting on test data
        predictions = nb_model.transform(test_data)
        #calculating accuracy
        evaluator = MulticlassClassificationEvaluator(labelCol='indexlabel',metricName="accuracy")
        acc = evaluator.evaluate(predictions) * 100
        accuracy.append(acc)
        text.insert(END,"Spark Naive Bayes Data Classifier Accuracy : "+str(acc)+"\n\n")
    except:
        traceback.print_exc() 


def runRandomForest():
    global X_train, X_test, y_train, y_test, rf
    dataset = pd.read_csv("weight.csv")
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    predict = rf.predict(X_test)
    random_acc = accuracy_score(y_test,predict) * 100
    text.insert(END,"Extension Random Forest Classifier Accuracy : "+str(random_acc)+"\n\n")
    accuracy.append(random_acc)

def runDecisionTree():
    global X_train, X_test, y_train, y_test
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    predict = dt.predict(X_test)
    dt_acc = accuracy_score(y_test,predict)* 100
    text.insert(END,"Extension Decision Tree Classifier Accuracy : "+str(dt_acc)+"\n\n")
    accuracy.append(dt_acc)

def runXGBoost():
    global X_train, X_test, y_train, y_test
    xg = XGBClassifier()
    xg.fit(X_train,y_train)
    predict = xg.predict(X_test)
    xg_acc = accuracy_score(y_test,predict) * 100
    text.insert(END,"Extension XGBoost Classifier Accuracy : "+str(xg_acc)+"\n\n")
    accuracy.append(xg_acc)
    

def dataAnalyzer():
    text.delete('1.0', END)
    global rf, word_meme
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=str(filename)+" Dataset Loaded")
    testData = pd.read_csv(filename)
    print(testData)
    for i in range(len(testData)):
        msg = testData.get_value(i, 'Tweets')
        tweet = str(msg)
        tweet = tweet.strip().lower()
        tweet = cleanPost(tweet)
        testReview = word_meme.transform([tweet]).toarray()
        print(testReview.shape)
        predict = rf.predict(testReview)
        print(predict.shape)
        predict = predict[0]
        if predict == 0:
            text.insert(END,"Tweet = "+str(msg)+"\n")
            text.insert(END,"PREDICTED AS =========> Garbage Tweet\n\n")
        if predict == 1:
            text.insert(END,"Tweet = "+str(msg)+"\n")
            text.insert(END,"PREDICTED AS =========> Advertisement Tweet\n\n")
        if predict == 2:
            text.insert(END,"Tweet = "+str(msg)+"\n")
            text.insert(END,"PREDICTED AS =========> Definite Tweet\n\n")    
        

def graph():
    global accuracy
    height = accuracy
    bars = ('Spark Naive Bayes','Extension Random Forest','Extension Decision Tree','Extension XGBoost')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms Comparison Graph")
    plt.show()

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Effective Garbage Data Filtering Algorithm for SNS Big Data Processing by Machine Learning')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload SNS Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

clsButton = Button(main, text="Data Classifier Generator", command=dataClassifier)
clsButton.place(x=50,y=150)
clsButton.config(font=font1)

sparkButton = Button(main, text="Data Classifier using SPARK Naive Bayes", command=naiveBayesTraining)
sparkButton.place(x=50,y=200)
sparkButton.config(font=font1)

rfButton = Button(main, text="Run Extension Random Forest", command=runRandomForest)
rfButton.place(x=50,y=250)
rfButton.config(font=font1)

dtButton = Button(main, text="Run Extension Decision Tree", command=runDecisionTree)
dtButton.place(x=50,y=300)
dtButton.config(font=font1)

xgboostButton = Button(main, text="Run Extension XGBoost", command=runXGBoost)
xgboostButton.place(x=50,y=350)
xgboostButton.config(font=font1)

analyzerButton = Button(main, text="Data Analyzer", command=dataAnalyzer)
analyzerButton.place(x=50,y=400)
analyzerButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=450)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=500)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=113)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
