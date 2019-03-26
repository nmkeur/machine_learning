from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


class Classifier():
    def __init__(self):
        self.skfold = ""

    def setupCV(self):
        n_folds=5
        n_repeats=10
        self.skfold = RepeatedStratifiedKFold(n_splits=n_folds,
                                              n_repeats=n_repeats,
                                              random_state=None)
