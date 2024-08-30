import numpy as np
import pandas as pd

df = pd.read_csv('D:\\NLP\\UPDATED_NLP_COURSE\\UPDATED_NLP_COURSE\\TextFiles\\smsspamcollection.tsv',sep='\t')
#print(df.head())
# print(df.isnull().sum())
print(df['label'].value_counts())
print(len(df))
## ham = 4825 , spam = 747 , total = 5572 


## preparing the train and test data sets
X=df['message']
y=df['label']

from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy = train_test_split(X,y,test_size = 0.33,random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
trainX_counts = count_vec.fit_transform(trainX)
print(trainX_counts.shape)


from sklearn.feature_extraction.text import TfidfTransformer
tf_idf = TfidfTransformer()
trainX_tfidf = tf_idf.fit_transform(trainX_counts)
print(trainX.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
trainX_combo = tfidfvec.fit_transform(trainX)
print(trainX_combo.shape)

from sklearn.svm import LinearSVC
lsvc_model = LinearSVC()
lsvc_model.fit(trainX_combo,trainy)

from sklearn.pipeline import Pipeline
text_clf  = Pipeline([('tfidfvec',TfidfVectorizer()),
                      ('lsvc_model',LinearSVC())])
text_clf.fit(trainX,trainy)
predictions = text_clf.predict(testX)

from sklearn import metrics
print(metrics.accuracy_score(testy,predictions))


