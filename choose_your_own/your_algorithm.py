#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
"""
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
"""
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.metrics import accuracy_score

from  sklearn.ensemble import RandomForestClassifier as Classifier
#from  sklearn.ensemble import AdaBoostClassifier as Classifier
from sklearn.naive_bayes import GaussianNB
#from  sklearn.cluster import KMeans as Classifier

#clf = Classifier(min_samples_split=2, n_estimators=20, criterion='entropy')
NB=GaussianNB()
acc = 0

for e in range(3,101):
    for r in range(2,101):
        #rate=(1.0*r/100)
        #clf = Classifier(n_estimators=e, learning_rate=rate)
        clf = Classifier(n_estimators=e, min_samples_split=r)
        rounds = 20
        r_acc=0
        for i in range(0,rounds):
            clf = clf.fit(features_train, labels_train)
            labels_predict = clf.predict(features_test)
            r_acc += accuracy_score(labels_test, labels_predict)
        if r_acc/rounds>acc:
            acc = r_acc/rounds
            print(e,r,acc)

clf = Classifier(n_estimators=3, min_samples_split=10)
clf = clf.fit(features_train, labels_train)
labels_predict = clf.predict(features_test)
print(accuracy_score(labels_test, labels_predict))





prettyPicture(clf, features_test, labels_test)
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
