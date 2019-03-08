# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class SVMclass():
    def __init__(self, kernell= None, cv= 5):
        self.x_train = ""
        self.y_train = ""
        self.x_test = ""
        self.y_test = ""
        self.kernell = kernell
        self.cv = cv
        print(self.kernell)

    def select_kernell():
        if kernell is "linear":
            print("Kernell is linear")
        elif kernell is "rbf":
            print("Kernell is rbf")
        elif kernell is "poly":
            print("kernell is poly")
        else:
            print("Unknown kernell")
            exit(0)
            
    def svc_param_selection(self):
        Cs = [0.0001,0.001, 0.01, 0.1, 1, 10]
        gammas = [0.0001,0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        svclassifier = SVC(kernel=self.kernell)
        grid_search = GridSearchCV(SVC(kernel=self.kernell), param_grid, cv=self.cv)
        grid_search.fit(self.x_train, self.y_train)
        grid_search.best_params_
        print(grid_search.best_params_, "JAJAJJAJJJAJJA")
        return grid_search.best_params_



        self.svc_param_selection()

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
