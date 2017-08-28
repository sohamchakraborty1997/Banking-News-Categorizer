import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, cross_validation, svm
from sklearn.preprocessing import MultiLabelBinarizer
from tkinter import *
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from textblob import TextBlob
import ngram
from tkinter import *
import os


dataset = pd.read_csv('data8.csv')


corpus = []
categories =[]


for i in range(0,258):
	data = dataset['headlines'][i]
	data = data.lower()
	corpus.append(data)
	categories.append(dataset['category'][i])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)
X_train_counts.shape


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

clf = OneVsRestClassifier(LinearSVC())
clf.fit(X_train_tfidf, categories)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train_tfidf, categories, test_size = 0.25)


clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = (accuracy_score(y_test,predicted))
accuracy = accuracy*100
print(accuracy)

docs_new = []
def result1():
	xtra_cat = []
	news = str(newsh.get())
	news = news.lower()
	output = news.lower()
	docs_new.append(output)
	rowsize = 5
	X_new_counts = count_vect.transform(docs_new)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	predicted = clf.predict(X_new_tfidf)
	for doc, category in zip(docs_new, predicted):
		labelresult = Label(myGUI, text="The given News Topic is => %s" % category.upper()).grid(row=rowsize,column=2)
		machinelearningoutput = category

###############Single category classification is fully accurate corresponding to the training data#####################
############The following approach of multicategory classification is not efficient at all , implemented due to shortage of time --This approach would be replaced with an efficient one in due time#########
	processed_news = []
	news = word_tokenize(news)
	for n in news:
		if n not in set(stopwords.words('english')):
			processed_news.append(n)

	proc_news = nltk.FreqDist(processed_news)
	word_features = list(proc_news.keys())
	processed_news = str(word_features)
	blob = TextBlob(processed_news)
	bigram = blob.ngrams(n=2)
	G = ngram.NGram(['home loans','car loans','education loans','debit cards','atms','student loans','automobile loans','housing loans'])
	for pair in bigram:
		pair = list(pair)
		if ((pair[0] == "'atm") or (pair[1] == "'atm")):
			pair = ['atms']
		pair = str(pair)
		#print(pair)
		frequency = G.search(pair,threshold=0.11)
		#print(frequency)
		count = 0
		for i,j in frequency:
			if count < 1:
				if i == 'atms':
					i = 'atm'
				xtra_cat.append(i)
				count = count + 1

	xtra_cat = set(xtra_cat)
	for x in xtra_cat:
				if ((x=="student loans") and (machinelearningoutput=="education loans")):
						return None
				if ((x=="automobile loans") and (machinelearningoutput=="car loans")):
						return None
				if ((x=="housing loans") and (machinelearningoutput=="home loans")):
						return None
				if ((x=="student loans")):
						x = "education loans"
				if ((x=="automobile loans")):
						x = "car loans"
				if ((x=="housing loans")):
						x = "home loans"
				if (x!= machinelearningoutput):
						rowsize = rowsize+2
						labelresult = Label(myGUI, text="The given News Topic is  => %s" % x.upper()).grid(row=rowsize,column=2)


myGUI = Tk()
myGUI.geometry('350x250')
myGUI.title('News Topic Classifier')

newsh=StringVar()

label1 = Label(myGUI, text='Welcome to News Topic Categorizer ',fg='red').grid(row=1,column=2)
#label2 = Label(myGUI,text="").grid(row=2,column=2)
label2 = Label(myGUI, text='Enter the News Below').grid(row=2,column=2)

S = Scrollbar(myGUI)
mynews = Entry(myGUI,textvariable=newsh, width=30).grid(row=4,column=2)
S.grid(row=4,column=0)

def menucat():
	return None	 


button1 = Button(myGUI,text='Predict',command=result1).grid(row=4,column=5)
#button2 = Button(myGUI,text='Reset',command=restart).grid(row=4,column=8)

menu = Menu(myGUI)
myGUI.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="New", command=menucat)
filemenu.add_command(label="Open...", command=menucat)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=myGUI.quit)
helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=menucat)


myGUI.mainloop()








