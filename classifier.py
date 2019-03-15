from sklearn.model_selection import RepeatedStratifiedKFold

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
class Classifier():
    def __init__(self):
        self.x_train = ""
        self.y_train = ""
        self.x_test = ""
        self.y_test = ""
        self.skfold = ""

    def setupCV(self):
        n_folds=3
        n_repeats=2
        self.skfold = RepeatedStratifiedKFold(n_splits=n_folds,
                                              n_repeats=n_repeats,
                                              random_state=None)
