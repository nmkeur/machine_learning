# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
import pandas as pd
import numpy as np
from classifier import Classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

class SVMsigmoid(Classifier):
    def __init__(self,x_train, y_train, x_test, y_test, kernell):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.kernell = kernell
        self.SVC_cls = ""
        self.y_pred = ""

        self.setupCV()
        self.svc_param_selection()


    def svc_param_selection(self):
        Cs = [0.01, 0.1, 1, 10]
        gammas = [1e-5,1e-4, 1e-3]
        param_grid = {'C': Cs, 'gamma' : gammas}
        self.SVC_cls = GridSearchCV(
                        SVC(kernel=self.kernell,random_state=1),
                        param_grid, cv=self.skfold,verbose=10,n_jobs=-1)
        self.SVC_cls.fit(self.x_train,self.y_train)

        print("Best parameters found on the training dataset:")
        print("Best value for C found: ", self.SVC_cls.best_params_.get('C'))
        print("Number value for Gamma found: ", self.SVC_cls.best_params_.get('gamma'))

        self.y_pred = self.SVC_cls.predict(self.x_test)

        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        print("Classification report for the test dataset:")
        print(classification_report(self.y_test, self.y_pred))





#svclassifierlinear = SVC(kernel='linear')
#svclassifierlinear.fit(x_train, y_train)

#svclassifiernonlinear = SVC(kernel='poly', degree=8)
#svclassifiernonlinear.fit(x_train, y_train)

#svclassifierrbf = SVC(kernel='rbf')
#svclassifierrbf.fit(x_train, y_train)

#svclassifiersigmoid = SVC(kernel='sigmoid',probability=True)
#svclassifiersigmoid.fit(x_train, y_train)

#y_pred = svclassifierlinear.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#y_pred = svclassifiernonlinear.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#y_pred = svclassifierrbf.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#y_pred = svclassifiersigmoid.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
