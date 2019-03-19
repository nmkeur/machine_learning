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
from CreatePlot import CreatePlot
from sklearn.model_selection import StratifiedKFold

class SVMlinear(Classifier):
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
        #for i in range (20):
        score = 'accuracy'
        #cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=i)
        #np.logspace(-10,10,num=21,base=10)
        #[1e-05,1e-04,1e-03,1e-02,0.1,1,10,100,1000]
        parameters = {'C': np.logspace(-1,3,num=5,base=10),# [500,1000,1500,2000]
        'gamma': np.logspace(-9,-3,num=11,base=10)}
        #'decision_function_shape':('ovo','ovr'),
        #'shrinking':(True,False)}

        SVC_cls = GridSearchCV(
        SVC(kernel='rbf',random_state=1,probability=True),
        parameters, cv=self.skfold, verbose=10,n_jobs=-1, scoring='f1')
        SVC_cls.fit(self.x_train,self.y_train)

        print("Best parameters found on the training dataset:")
        print("Best value for C found: ", SVC_cls.best_params_.get('C'))
        print("Number value for Gamma found: ", SVC_cls.best_params_.get('gamma'))

        y_pred = SVC_cls.predict(self.x_test)

        print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))
        print("Classification report for the test dataset:")
        print(classification_report(self.y_test, y_pred))

        CP = CreatePlot()
        #CP.plot_confusion_matrix(self.y_test, y_pred)
        #CP.plot_precision_recall_curve(SVC_cls, self.x_test , self.y_test)
        #CP.plot_roc_curve(SVC_cls, self.x_test , self.y_test)
        #print(SVC_cls.cv_results_)
        CP.plot_grid_search(SVC_cls.cv_results_, parameters.get('gamma'),
        parameters.get('C'), 'gamma', 'C', True)
        #rf = SVC(kernel=self.kernell, gamma=0.001,random_state=1)
        #CP.plot_learning_curve(rf, self.x_train , self.y_train, self.skfold)
        # Calling Method
        #CP.plot_grid_search(self.RF_clf.cv_results_, parameters.get('n_estimators'),parameters.get('max_features') ,
        #                 'N Estimators', 'Max Features')

        print("# Tuning hyper-parameters for %s" % score)

        print("Best parameters set found on development set:")
        print()
        print(SVC_cls.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = SVC_cls.cv_results_['mean_test_score']
        stds = SVC_cls.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, SVC_cls.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        #y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(self.y_test, y_pred))
        print()

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
