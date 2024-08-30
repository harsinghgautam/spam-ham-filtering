## classifying text messages as spam or ham based on length of text and number of punctuations
## comparative study of models
## employed 3 ML models [ logistic regression, naive-bayes classifier, and SVC(support vector classification)]
## mentioned in the order of increasing accuracy


import numpy as np
import pandas as pd

myfile = pd.read_csv('D:\\NLP\\UPDATED_NLP_COURSE\\UPDATED_NLP_COURSE\\TextFiles\\smsspamcollection.tsv',sep='\t')
# print(myfile.head())
# print(myfile.isnull().sum())
# print(myfile['label'].unique())
# print(myfile['label'].value_counts())
print(type(myfile))
# print(myfile['length'].describe())
#highly skewed data mean is 80.5 and max is 910!
#plotting on normal axis system is not preferable
#plotting on logarithmic axis

import matplotlib.pyplot as plt

# plt.xscale('log')
# bins = 1.15**np.arange(0,50) #1.15^49 = 942
# plt.hist(myfile[myfile['label']=='ham']['length'],bins=bins,alpha=0.8)
# plt.hist(myfile[myfile['label']=='spam']['length'],bins=bins,alpha=0.8)
# plt.legend(('ham','spam'))
# plt.xlabel('log of length')
# plt.ylabel('frequency')
# plt.show()

## punctuation analysis
# print(myfile['punct'].describe())
# bins2 = 1.5**np.arange(0,15)
# plt.xscale('log')
# plt.hist(myfile[myfile['label'=='ham']]['punct'],bins=bins2,alpha=0.8)
# plt.hist(myfile[myfile['label'=='spam']]['punct'],bins=bins2,alpha=0.8)
# plt.xlabel('log of punct freq')
# plt.ylabel('freq')
# plt.legend('ham','spam')
# plt.show()

# plt.xscale('log')
# bins = 1.5**(np.arange(0,15))
# print(bins)
# plt.hist(myfile[myfile['label']=='ham']['punct'],bins=bins,alpha=0.8)
# plt.hist(myfile[myfile['label']=='spam']['punct'],bins=bins,alpha=0.8)
# plt.legend(('ham','spam'))
# plt.show()

X = myfile[['length','punct']]
y = myfile['label']
from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy = train_test_split(X,y,test_size = 0.33,random_state = 42)
print('Training Data Shape:', trainX.shape)
print('Testing Data Shape: ', testX.shape)


# ## LOGISTIC REGRESSION MODEL
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(solver = 'lbfgs')
# lr_model.fit(trainX,trainy)

# from sklearn import metrics
# predictions = lr_model.predict(testX)
# print(metrics.confusion_matrix(testy,predictions))
# print(metrics.classification_report(testy,predictions))
# print('accuracy score = ',metrics.accuracy_score(testy,predictions))

# ## Naive bayes classifier
# from sklearn.naive_bayes import MultinomialNB
# nb_model = MultinomialNB()
# nb_model.fit(trainX,trainy)

# from sklearn import metrics as m
# predictions = nb_model.predict(testX)
# print(m.confusion_matrix(testy,predictions))
# print('accuracy score = ',m.accuracy_score(testy,predictions))
# print(m.classification_report(testy,predictions))


## SVM - support vector machine / SVC - support vector classifier
from sklearn.svm import SVC
svc_model = SVC(gamma='auto')
svc_model.fit(trainX,trainy)
predictions = svc_model.predict(testX)

from sklearn import metrics as m
print(m.confusion_matrix(testy,predictions))
print('accuracy = ',m.accuracy_score(testy,predictions))
