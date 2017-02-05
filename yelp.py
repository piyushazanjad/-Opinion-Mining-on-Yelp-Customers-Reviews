from Tkinter import *
from tkMessageBox import *
import tkFileDialog as filedialog
from tkFileDialog import askopenfilename
import json
import nltk
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from nltk.corpus import stopwords
import plotly
import plotly.plotly as py 
py.sign_in('', '')
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly import tools
import time

root = Tk()

text1 = Text(root, height=20, width=35)
photo=PhotoImage(file="logo.gif")
text1.insert(END,'\n')
text1.image_create(END, image=photo, align='baseline')

text1.pack(side=LEFT)

text2 = Text(root, height=20, width=55)
scroll = Scrollbar(root, command=text2.yview)
text2.configure(yscrollcommand=scroll.set)
text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
text2.tag_configure('big', font=('Verdana', 20, 'bold'))
text2.tag_configure('color', foreground='#476042', 
font=('Tempus Sans ITC', 12, 'bold'))
text2.tag_bind('follow', '<1>', lambda e, t=text2: t.insert(END, "Not now, maybe later!"))
text2.insert(END,'\nCPSC 531 - Project\n', 'big')
text2.insert(END,'\nOPINION MINING ON YELP CUSTOMERS REVIEWS\n', 'big')
quote = """
This project uses SVM classification technique to classify 
the Yelp dataset of customer reviews. The SVM is
implemented using Python NLP.
"""
text2.insert(END, quote, 'color')
text2.pack(side=LEFT)
scroll.pack(side=RIGHT, fill=Y)

def OpenFile():
	name = askopenfilename()
	print (name)
	f=open(name)
	line=f.readline()
	dictSent={}
	dictStar={}
	reviewList=[]
	i=1
	sentiment='negative'
	while line:
		line=f.readline()
		review=json.loads(line)
		index=i
		star=review["stars"]
		text= review["text"]
		if star>3:
			sentiment='1.0' #positive
		else:
			sentiment='0.0'  #negative
		dictSent[index]=sentiment
		dictStar[index]=star
		reviewList.append([index,text])
		print(i);
		i+=1
		if i==101:
			break
	f.close()
	print "Dataset Loaded..."
	tokenizedWords=tokenizeReview(reviewList)
	print "Reviews Tokenized..."
	print "#"*70
	print"\n Classification without any processing"
	print "#"*70
	lexicon,tfVector=createTfIdfMatrix(tokenizedWords)
	print "TF Matrix Created..."
	print "length of vector : ",len(tfVector[1])
	tags=createTags(dictSent)
	trainVecs=np.array(tfVector.values())
	trainTags=np.array(tags)
	classification(trainVecs,trainTags,2,'result1')
	print"#"*70
	print"\n Classification after removing stop words"
	print "#"*70
	tokenizedWords=removeStopwords(tokenizedWords)
	lexicon,tfVector=createTfIdfMatrix(tokenizedWords)
	print "TF Matrix Created..."
	print "length of vector : ",len(tfVector[1])
	tags=createTags(dictSent)
	trainVecs=np.array(tfVector.values())
	trainTags=np.array(tags)
	classification(trainVecs,trainTags,2,'result2')
	time.sleep(20)
	print"#"*70
	print "\n Classification into 5 Classes"
	print"#"*70
	tags=createTags(dictStar)
	trainTags=np.array(tags)
	classification(trainVecs,trainTags,5,'result3')

def About():
	quote = """
	\nCPSC 531 - Project OPINION MINING ON YELP CUSTOMERS REVIEWS\n"
	This project uses SVM classification technique to classify the Yelp dataset of customer reviews. The SVM is implemented using Python NLP.
	\nGuided By:
	Professor: Dr. Chun-l Phillip Chen
	\nSubmitted By:
	Piyusha Zanjad(CWID: 893578021))
	Gargi Mrunal Kulkarni(CWID: 893210922))
	"""
	showinfo('About', quote)
	
menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Open...", command=OpenFile)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

helpmenu = Menu(menu)
menu.add_cascade(label="About", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

def tokenizeReview(reviewList):
	tokenizedWords={}
	for review in reviewList:

		tokenizedWords[review[0]]=word_tokenize(review[1])
		#print tokenizedWords
	return tokenizedWords 

def buildLexicon(tokenizedWords):
	lexicon=set()
	i=1
	for i in range(1,len(tokenizedWords)+1):
		lexicon.update(tokenizedWords[i])
	return lexicon

def tf(word,tokenizedWords):
	return tokenizedWords.count(word)

def createTfIdfMatrix(tokenizedWords):
	lexicon=buildLexicon(tokenizedWords)
	tf_vector={}
	for i in range(1, len(tokenizedWords)+1):
		tf_vector[i]=[tf(word,tokenizedWords[i]) for word in lexicon]
	return lexicon,tf_vector

def createTags(dictSent):
	tags=dictSent.values()
	return tags

def classification(trainVecs,trainTags,n, file):
	clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False))
	clf.fit(trainVecs, trainTags)
	print "Classifier Trained..."
	predicted = cross_validation.cross_val_predict(clf, trainVecs, trainTags, cv=5)
	accuracy = metrics.accuracy_score(trainTags, predicted)
	precision = metrics.precision_score(trainTags, predicted,pos_label=None,average='weighted')
	recall = metrics.recall_score(trainTags, predicted,pos_label=None,average='weighted')
	print "Cross Fold Validation Done..."
	print "accuracy score: ", metrics.accuracy_score(trainTags, predicted)
	print "precision score: ", metrics.precision_score(trainTags, predicted,pos_label=None,average='weighted')
	print "recall score: ", metrics.recall_score(trainTags, predicted,pos_label=None,average='weighted')
	print "classification_report: \n ", metrics.classification_report(trainTags, predicted)
	print "confusion_matrix:\n ", metrics.confusion_matrix(trainTags, predicted)
	support=support_classification_report(metrics.classification_report(trainTags, predicted))
	plotGraph(accuracy,precision,recall,support,n,file)
	return

def plotGraph(accuracy,precision,recall,support,n,file):
	trace1 = go.Bar(
		x=['Accuracy', 'Precision', 'Recall'],
		y=[accuracy, precision, recall]
	)
	if n==2:
		trace2 = go.Bar(
			x=['Class 0','Class 1'],
			y=[support[0],support[1]]
		)
	else:
		trace2 = go.Bar(
			x=['Class 1','Class 2','Class 3','Class 4','Class 5'],
			y=[support[0],support[1],support[2],support[3],support[4]]
		)
	fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Metrics', 'Classification'))
	fig.append_trace(trace1, 1, 1)
	fig.append_trace(trace2, 1, 2)
	fig['layout'].update(height=600, width=600, title='Results')
	plot_url = py.plot(fig, filename='make-subplots-multiple-with-title Grid')
	return

def removeStopwords(tokenizedWords):
	for i in range(1,len(tokenizedWords)+1):
		filteredWords=[word for word in tokenizedWords[i] if word not in stopwords.words('english')]
		tokenizedWords[i]=filteredWords
	return tokenizedWords

def support_classification_report(classification_report):

	lines = classification_report.split('\n')
	support = []
	for line in lines[2 : (len(lines) - 2)]:
		t = line.strip().split()
		if len(t) < 2: continue
		v = [float(x) for x in t[1: len(t) - 1]]
		support.append(int(t[-1]))
	#print('support: {0}'.format(support))
	return support

mainloop()