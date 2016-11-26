#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

print labels_test

#########################################################
### your code goes here ###
#from sklearn.naive_bayes import GaussianNB as nb
from sklearn import svm
from sklearn.metrics import accuracy_score
#clf = nb()
c = 10000
print "C", c
clf = svm.SVC(kernel='rbf', C=c)
t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"
t0 = time()

labels_predict = clf.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s"
print "Accuracy:", accuracy_score(labels_test, labels_predict)
print clf.predict(features_test[10])
print clf.predict(features_test[26])
print clf.predict(features_test[50])


#########################################################


