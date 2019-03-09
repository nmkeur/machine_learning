class SVMClassifier():
    def __init__(self):
        self.x_train = ""
        self.y_train = ""
        self.x_test = ""
        self.y_test = ""
        self.kernell = kernell
        self.cv = cv
        self.select_kernell()
