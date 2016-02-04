#!/usr/bin/env python


# ---------------------------------------Imports------------------------------------------------------------------------

import os
import re
import nltk
import glob
import pickle
import datetime
import copy

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import svm
# ---------------------------------------Constants----------------------------------------------------------------------

POS_REVIEW_FOLDER = r"C:\Ofir\Tau\Machine Learning\Project\project\pos"
NEG_REVIEW_FOLDER = r"C:\Ofir\Tau\Machine Learning\Project\project\neg"

# ---------------------------------------Classes------------------------------------------------------------------------

class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences


# ---------------------------------------Functions----------------------------------------------------------------------


# ---------------------------------------Data extraction----------------------------------------------------------------

def load_imdb_data():
    pos_reviews = glob.glob(POS_REVIEW_FOLDER + r"\*")
    neg_reviews = glob.glob(NEG_REVIEW_FOLDER + r"\*")
    all_reviews = pos_reviews + neg_reviews
    reviews = pd.DataFrame(columns=["review_id", "review", "sentiment"])
    for i in range(len(all_reviews)):
        if ((i+1)%100 == 0):
            print "processed %d reviews" % i
        review_file = all_reviews[i]
        sentiment = 1 if "pos" in review_file else -1
        with open(review_file, "r") as fh:
            review_text = fh.read()
        reviews.loc[i] = [i, review_text, sentiment]
    return reviews


def get_clean_reviews(reviews):
    """

    :param reviews: a DataFrame with 3 columns review_id, review, score
    :type reviews: pandas.DataFrame
    :return:
    """
    clean_reviews_df = pd.DataFrame(columns=["review_id", "review", "sentiment"])
    for index, row in reviews.iterrows():
        row["review"] = KaggleWord2VecUtility.review_to_wordlist( row["review"], remove_stopwords=True )
        clean_reviews_df.loc[index] = row
    return clean_reviews_df

# ---------------------------------------Feature creation - average vector----------------------------------------------

def makeFeatureVec(words, model, index2word_set, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #

    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #

    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000. == 0.:
       #    print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, index2word_set, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def sentiment_analysis_random_forest(train, test, word2vec_model, num_features, num_estimators):
    trainDataVecs = getAvgFeatureVecs(train["review"], word2vec_model, num_features)
    testDataVecs = getAvgFeatureVecs(test["review"], word2vec_model, num_features)
    #forest = RandomForestClassifier(n_estimators=num_estimators)
    forest = GradientBoostingClassifier(n_estimators=num_estimators)
    forest.fit(trainDataVecs, train["sentiment"])
    result = forest.score(testDataVecs, test["sentiment"])
    print result
    return result

def sentiment_analysis_svm(train, test, word2vec_model, num_features):
    trainDataVecs = getAvgFeatureVecs(train["review"], word2vec_model, num_features)
    testDataVecs = getAvgFeatureVecs(test["review"], word2vec_model, num_features)
    clf = svm.SVC(C=1)
    clf.fit(trainDataVecs, train["sentiment"])
    result = clf.score(testDataVecs, test["sentiment"])
    print result
    return result


def cross_validation(reviews, word2vec_model, num_features, num_estimators, test_size=0.1):
    num_samples = len(reviews)
    perm = np.random.permutation(num_samples).tolist()
    assert 0 < test_size < 1, "test size is a value between 0 and 1."
    num_of_iterations = int(1 / test_size)
    score = 0
    for i in range(num_of_iterations):
        print "Cross validation iter #%d" % (i+1)
        test_indexes = perm[int(i*test_size*num_samples):int((i+1)*test_size*num_samples)]
        train_indexes = perm[:int(i*test_size*num_samples)] + perm[int((i+1)*test_size*num_samples):]
        test_data = reviews.loc[test_indexes]
        train_data = reviews.loc[train_indexes]
        score += sentiment_analysis_random_forest(train_data, test_data, word2vec_model, num_features, num_estimators=num_estimators)
        #score += sentiment_analysis_svm(train_data, test_data, word2vec_model, num_features)
    print score / num_of_iterations


# ---------------------------------------Feature creation - K-Means-----------------------------------------------------

def create_word2vec_cluster(word2vec_model):
    word_vectors = word2vec_model.syn0
    num_clusters = word_vectors.shape[0] / 1000
    spectral_cluster_model = SpectralClustering(n_clusters=num_clusters)
    idx = spectral_cluster_model.fit_predict(word_vectors)
    pickle.dump(spectral_cluster_model, open(r"C:\Ofir\Tau\Machine Learning\Project\project\k_means_model.pkl", "wb"))
    return spectral_cluster_model





if __name__ == "__main__":

    reviews_csv = r"C:\Ofir\Tau\Machine Learning\Project\project\all_reviews.csv"
    clean_reviews_csv = r"C:\Ofir\Tau\Machine Learning\Project\project\all_clean_reviews.csv"
    word2vec_model_fp = r"C:\Ofir\Tau\Machine Learning\Project\word2vec\Dictionary\GoogleNews-vectors-negative300.bin"
    num_features = 300
    compute_reviews = False
    compute_clean_reviews = False
    do_average_vec_cv = True
    do_k_means_clustering = False

    if compute_reviews:
        reviews = load_imdb_data()
        reviews.to_csv(reviews_csv)
    else:
        reviews = pd.read_csv(reviews_csv)

    if compute_clean_reviews:
        clean_reviews = get_clean_reviews(reviews)
        clean_reviews.to_csv(clean_reviews_csv)
    else:
        clean_reviews = pd.read_csv(clean_reviews_csv)

    word2vec_model = Word2Vec.load_word2vec_format(word2vec_model_fp, binary=True)
    # word2vec_model.init_sims(replace=True)
    if do_average_vec_cv:
        print "Started cross validation..."
        cross_validation(clean_reviews, word2vec_model, num_features, num_estimators=100)
    if do_k_means_clustering:
        now = datetime.datetime.now()
        create_word2vec_cluster(word2vec_model)
        print "time took to make k means clustering: %s" % (datetime.datetime.now() - now).__str__()



