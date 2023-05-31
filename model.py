# Useful machine-learning commands
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


class LogisticRegression(object):
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear', random_state=42)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def coef(self):
        return self.model.coef_

    def score(self, X, y):
        return self.model.score(X, y)

    def error(self, X, y):
        return sum(np.where(self.model.predict(X) == y, 1, 0) == 0) / len(X)


class SGDClassifier(object):
    def __init__(self, params):
        self.model = SGDClassifier(loss='log', learning_rate='optimal',
                                   alpha=params['alpha'], penalty=params['penalty'],
                                   max_iter=params['max_iter'], tol=params['tol'], shuffle=True, random_state=100)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def logloss(self, X, y, epochs: int):
        epochLoss = []
        for _ in range(epochs):
            self.model.partial_fit(X, y, classes=np.unique(y))
            epochLoss.append(log_loss(y, self.model.predict(X)))
        return epochLoss
