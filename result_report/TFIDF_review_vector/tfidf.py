
import pandas as pd
import numpy as np
import pickle
import random

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def knn(X_train, X_test, Y_train, Y_test):
    n_neighbors = [5, 8, 15]
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    weights = ['uniform', 'distance']
    for nn in n_neighbors:
        for algo in algorithm:
            for w in weights:
                model = KNeighborsClassifier(n_neighbors=nn, weights=w, algorithm=algo)
                model.fit(X_train, Y_train)
                score = model.score(X_test, Y_test)
                print "KNeighborsClassifier(n_neighbors=%d, weights='%s', algorithm='%s') -> %.4f" % (nn, w, algo, score)


def random_forest(X_train, X_test, Y_train, Y_test):
    estimators = [10, 100, 500]
    criterion = ["gini", "entropy"]
    max_features = ["auto", "sqrt", "log2"]
    for est in estimators:
        for cr in criterion:
            for mf in max_features:
                forest = RandomForestClassifier(n_estimators=est,
                                                n_jobs=8,
                                                criterion=cr,
                                                max_features=mf)
                forest.fit(X_train, Y_train)
                score = forest.score(X_test, Y_test)
                print "RandomForestClassifier(n_estimators=%s, n_jobs=8, criterion=%s, max_features=%s) -> %.4f" % (est,
                                                                                                                    cr,
                                                                                                                    mf,
                                                                                                                    score)

def extreme_tree(X_train, X_test, Y_train, Y_test):
    estimators = [10, 100, 500]
    criterion = ["gini", "entropy"]
    max_features = ["auto", "sqrt", "log2"]
    for est in estimators:
        for cr in criterion:
            for mf in max_features:
                extre_model = ExtraTreesClassifier(n_jobs=8,
                                                   random_state=np.random.RandomState(),
                                                   n_estimators=est,
                                                   criterion=cr,
                                                   max_features=mf)
                extre_model.fit(X_train, Y_train)
                score = extre_model.score(X_test, Y_test)
                print "ExtraTreesClassifier(n_jobs=8, random_state=np.random.RandomState(), n_estimators=%d, criterion=%s, max_features=%s) -> %.4f" % (est, cr, mf, score)


def multinomial_nb(X_train, X_test, Y_train, Y_test):
    alphas = [1, 10, 100, 1000]
    fit_prior = [True, False]
    for alpha in alphas:
        for fp in fit_prior:
            multi_nb = MultinomialNB(alpha=alpha, fit_prior=fp)
            multi_nb.fit(X_train, Y_train)
            score = multi_nb.score(X_test, Y_test)
            print "MultinomialNB(alpha=%d, fit_prior=%s) -> %.4f" % (alpha, str(fp), score)


def gaussian_nb(X_train, X_test, Y_train, Y_test):
    gaus_nb = GaussianNB()
    gaus_nb.fit(X_train, Y_train)
    score = gaus_nb.score(X_test, Y_test)
    print "GaussianNB() -> %.4f" % score


def cross_validate(X, Y, iters, model, params):
    """

    :param X: the data
    :param Y: labels
    :param iters: number of cross validation iterations
    :param model: the classifier model
    :param params: parameters to initialize the model
    :type params: dict
    :type iters: int
    :return:
    """
    avg_score = 0
    for i in range(iters):
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=random.randint(0, 10**9))
        classifier = model(**params)
        classifier.fit(X_train, Y_train)
        score = classifier.score(X_test, Y_test)
        avg_score += score
        print "Cross Validation iter #%d: %.4f" % (i+1, score)
    print "Cross Validation average accuracy: %.4f" % (avg_score/iters)


if __name__ == "__main__":
    all_reviews = pd.read_csv(r"C:\Ofir\Tau\Machine Learning\Project\project\all_reviews.csv")
    all_clean_reviews = pd.read_csv(r"C:\Ofir\Tau\Machine Learning\Project\project\all_clean_reviews.csv")
    for i in range(300,2100,150):
        print "##### TFIDF #features = %d" % i
        vectorizer = TfidfVectorizer(analyzer='word',
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=i)
        train_data_features = vectorizer.fit_transform(all_clean_reviews["review"])
        train_data_features = train_data_features.toarray()
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(train_data_features,
                                                                             all_clean_reviews["sentiment"],
                                                                             test_size=0.1,
                                                                             random_state=141)
        extreme_tree(X_train, X_test, Y_train, Y_test)
        random_forest(X_train, X_test, Y_train, Y_test)
        multinomial_nb(X_train, X_test, Y_train, Y_test)
        gaussian_nb(X_train, X_test, Y_train, Y_test)
    # knn(X_train, X_test, Y_train, Y_test)



    """
    cross_validate(X=train_data_features,
                   Y=all_clean_reviews["sentiment"],
                   iters=10,
                   model=RandomForestClassifier,
                   params={"n_estimators": 1000,
                           "n_jobs": 8,
                           "criterion": "entropy",
                           "max_features": "log2"})
    """
