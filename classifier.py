

import scipy
from scipy import sparse
import numpy as np
from collections import Counter
import string
from tqdm import tqdm
import math
from sklearn.feature_extraction.text import CountVectorizer
# utility functions we provides

def load_data(file_name):
    '''
    @input:
     file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
     a list of sentences
    '''
    with open(file_name, "r") as file:
        sentences = file.readlines()
    return sentences


def tokenize(sentence):
    # Convert a sentence into a list of words
    wordlist = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(
        ' ')

    return [word.strip() for word in wordlist]


# Main "Feature Extractor" class:
# It takes the provided tokenizer and vocab as an input.

class feature_extractor:
    def __init__(self, vocab, tokenizer):
        self.tokenize = tokenizer
        self.vocab = vocab  # This is a list of words in vocabulary
        self.vocab_dict = {item: i for i, item in
                           enumerate(vocab)}  # This constructs a word 2 index dictionary
        self.d = len(vocab)

    def bag_of_word_feature(self, sentence):
        '''
        Bag of word feature extactor
        :param sentence: A text string representing one "movie review"
        :return: The feature vector in the form of a "sparse.csc_array" with shape = (d,1)
        '''

        # TODO ======================== YOUR CODE HERE =====================================
        # Hint 1:  there are multiple ways of instantiating a sparse csc matrix.
        #  Do NOT construct a dense numpy.array then convert to sparse.csc_array. That will defeat its purpose.

        # Hint 2:  There might be words from the input sentence not in the vocab_dict when we try to use this.

        # Hint 3:  Python's standard library: Collections.Counter might be useful
        # Split the sentence into words
        words = sentence.split()

        # Count the occurrence of each word
        word_counts = Counter(words)

        # Create a list to hold the data and row indices for the sparse matrix
        data = []
        row_indices = []

        # Iterate over the vocabulary
        for word, index in self.vocab_dict.items():
            if word in word_counts:
                # If the word is in the sentence, add its count to the data and its index to the row indices
                data.append(word_counts[word])
                row_indices.append(index)

        # Create a sparse matrix from the data and row indices
        x = sparse.csc_matrix((data, (row_indices, [0]*len(data))), shape=(len(self.vocab_dict), 1))

        # x = 0
        # TODO =============================================================================
        return x


    def __call__(self, sentence):
        # This function makes this any instance of this python class a callable object
        return self.bag_of_word_feature(sentence)


class classifier_agent():
    def __init__(self, feat_map, params):
        '''
        This is a constructor of the 'classifier_agent' class. Please do not modify.

         - 'feat_map'  is a function that takes the raw data sentence and convert it
         into a data vector compatible with numpy.array

         Once you implement Bag Of Word and TF-IDF, you can pass an instantiated object
          of these class into this classifier agent

         - 'params' is an numpy array that describes the parameters of the model.
          In a linear classifer, this is the coefficient vector. This can be a zero-initialization
          if the classifier is not trained, but you need to make sure that the dimension is correct.
        '''
        self.feat_map = feat_map
        self.params = np.array(params)

    def batch_feat_map(self, sentences):
        '''
        This function processes data according to your feat_map. Please do not modify.

        :param sentences:  A single text string or a list of text string
        :return: the resulting feature matrix in sparse.csc_array of shape d by m
        '''
        if isinstance(sentences, list):
            X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in sentences])
            # X = np.vstack([self.feat_map(sentence) for sentence in sentences])
        else:
            X = self.feat_map(sentences)
        return X

    def score_function(self, X):
        '''
        This function computes the score function of the classifier.
        Note that the score function is linear in X
        :param X: A scipy.sparse.csc_array of size d by m, each column denotes one feature vector
        :return: A numpy.array of length m with the score computed for each data point
        '''

        (d,m) = X.shape
        s = np.zeros(shape=m) # this is the desired type and shape for the output
        # TODO ======================== YOUR CODE HERE =====================================
        s = X.T.dot(self.params).flatten()
        # TODO =============================================================================
        return s



    def predict(self, X, RAW_TEXT=False, RETURN_SCORE=False):
        '''
        This function makes a binary prediction or a numerical score
        :param X: d by m sparse (csc_array) matrix
        :param RAW_TEXT: if True, then X is a list of text string
        :param RETURN_SCORE: If True, then return the score directly
        :return:
        '''
        if RAW_TEXT:
            X = self.batch_feat_map(X)

        # TODO ======================== YOUR CODE HERE =====================================
        # This should be a simple but useful function.
        preds = np.zeros(shape=X.shape[1])
        # Compute the score for each data point
        scores = self.score_function(X)

        if RETURN_SCORE:
            return scores

        # Make binary predictions
        preds = np.where(scores >= 0, 1, -1)
        # TODO =============================================================================

        return preds


    def error(self, X, y, RAW_TEXT=False):
        '''
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :param RAW_TEXT: if True, then X is a list of text string,
                        and y is a list of true labels
        :return: The average error rate
        '''
        if RAW_TEXT:
            X = self.batch_feat_map(X)
            y = np.array(y)

        # TODO ======================== YOUR CODE HERE =====================================
        # The function should work for any integer m > 0.
        # You may wish to use self.predict
        err =  0.0
        # Make predictions
        preds = self.predict(X)

        # Compute the error rate
        err = np.mean(preds != y)
        # TODO =============================================================================

        return err


    def loss_function(self, X, y):
        '''
        This function implements the logistic loss at the current self.params

        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return:  a scalar, which denotes the mean of the loss functions on the m data points.

        '''

        # TODO ======================== YOUR CODE HERE =====================================
        # The function should work for any integer m > 0.
        # You may first call score_function

        # loss =  0.0
        # Compute the score for each data point
        scores = self.score_function(X)
        scores= np.clip(scores, -700, 700)

        # Compute the logistic loss for each data point
        losses = np.log(1 + np.exp(-y * scores))

        # Compute the mean of the losses
        loss = np.mean(losses)

        # TODO =============================================================================

        return loss

    def gradient(self, X, y):
        '''
        It returns the gradient of the (average) loss function at the current params.
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return: Return an nd.array of size the same as self.params
        '''

        # TODO ======================== YOUR CODE HERE =====================================
        # Hint 1:  Use the score_function first
        # Hint 2:  vectorized operations will be orders of magnitudely faster than a for loop
        # Hint 3:  don't make X a dense matrix

        # grad = np.zeros_like(self.params)
        # Compute the score for each data point
        scores = self.score_function(X)

        y = y.reshape(1, -1)
        # product = -X.multiply(y)
        # print("Shape of -X.multiply(y_column):", product.shape)
        # print("Shape of 1 + np.exp(y * scores):", (1 + np.exp(y * scores)).shape)

        gradient = -X.multiply(y) / (1 + np.exp(y * scores))
        # Compute the mean of the gradients
        gradient = np.mean(gradient, axis=1)

        # TODO =============================================================================
        return gradient


    def train_gd(self, train_sentences, train_labels, niter, lr=0.01):
        '''
        The function should updates the parameters of the model for niter iterations using Gradient Descent
        It returns the sequence of loss functions and the sequence of training error for each iteration.

        :param train_sentences: Training data, a list of text strings
        :param train_labels: Training data, a list of labels 0 or 1
        :param niter: number of iterations to train with Gradient Descent
        :param lr: Choice of learning rate (default to 0.01, but feel free to tweak it)
        :return: A list of loss values, and a list of training errors.
                (Both of them has length niter + 1)
        '''

        Xtrain = self.batch_feat_map(train_sentences)
        ytrain = np.array(train_labels)
        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]
        # TODO ======================== YOUR CODE HERE =====================================
        # You need to iteratively update self.params
        for _ in tqdm(range(niter), desc="Training progress"):
            # Compute the gradient
            gradient = self.gradient(Xtrain, ytrain)
            # gradient_mean = np.mean(gradient, axis=0)
            gradient_flat = np.ravel(gradient)

            # Update the weights
            # self.params -= lr * gradient_mean
            self.params -= lr * gradient_flat


            # Compute the loss and error for the new weights
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))
        # TODO =============================================================================
        return train_losses, train_errors


    def train_sgd(self, train_sentences, train_labels, nepoch, lr=0.001):
        '''
        The function should updates the parameters of the model for using Stochastic Gradient Descent.
        (random sample in every iteration, without minibatches,
        pls follow the algorithm from the lecture).

        :param train_sentences: Training data, a list of text strings
        :param train_labels: Training data, a list of labels 0 or 1
        :param nepoch: Number of effective data passes.  One data pass is the same as n iterations
        :param lr: Choice of learning rate (default to 0.001, but feel free to tweak it)
        :return: A list of loss values and a list of training errors.
                (initial loss / error plus  loss / error after every epoch, thus length epoch +1)
        '''



        Xtrain = self.batch_feat_map(train_sentences)
        ytrain = np.array(train_labels)
        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]
        # TODO ======================== YOUR CODE HERE =====================================
        # You need to iteratively update self.params
        # You should use the following for selecting the index of one random data point.

        # idx = np.random.choice(len(ytrain), 1)
        for _ in range(nepoch):
            # Select a random data point
            idx = np.random.choice(len(ytrain), 1)

            # Compute the gradient for the selected data point
            gradient = self.gradient(Xtrain[:, idx], ytrain[idx])
            gradient_flat = np.ravel(gradient)
            # Update the weights
            self.params -= lr * gradient_flat

            # Compute the loss and error for the new weights
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))


        # TODO =============================================================================
        return train_losses, train_errors


    def eval_model(self, test_sentences, test_labels):
        '''
        This function evaluates the classifier agent via new labeled examples.
        Do not edit please.
        :param train_sentences: Training data, a list of text strings
        :param train_labels: Training data, a list of labels 0 or 1
        :return: error rate on the input dataset
        '''
        X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in test_sentences])
        y = np.array(test_labels)
        return self.error(X, y)

    def save_params_to_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_params_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)



class tfidf_extractor(feature_extractor):
    '''
    This class gives you an example how to write a customer feature extractor
    '''
    def __init__(self, vocab, tokenizer, word_idf):
        super().__init__(vocab, tokenizer)
        self.word_idf = word_idf
        # self.tokenizer = tokenizer()

    def tfidf_feature(self,sentence):
        # -------- Your implementation of the tf-idf feature ---------------
        # TODO ======================== YOUR CODE HERE =====================================
        # x = self.bag_of_word_feature(sentence)
        # Split the sentence into words
        words = tokenize(sentence)

        # Initialize a dictionary to hold the word frequencies
        word_freqs = Counter(words)

        # Initialize a dictionary to hold the tf-idf values
        tfidf_values = {}

        # Calculate the tf-idf value for each word
        for word, freq in word_freqs.items():
            if word in self.vocab:
                tfidf_values[word] = freq * self.word_idf.get(word, 0)
                
        # Convert self.vocab to a dictionary if it's a list
        if isinstance(self.vocab, list):
            self.vocab = {word: index for index, word in enumerate(self.vocab)}

        # Create a sparse matrix from the tf-idf values
        row_indices = [self.vocab[word] for word in tfidf_values if word in self.vocab]
        data = list(tfidf_values.values())
        x = sparse.csc_matrix((data, (row_indices, [0]*len(data))), shape=(self.d, 1))


        # TODO =============================================================================
        return x

    def __call__(self, sentence):
        return self.tfidf_feature(sentence)


def compute_word_idf(train_sentences,vocab):
    '''
    This function computes the inverse document frequency of words in vocab.
    See: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    Please use natural log whenever you encounter logarithm.

    :param train_sentences: Training data
    :param vocab: A list of words
    :return: A dictionary of the format {word_1: idf_1, ..., word_d:idf_d}
    according to the same order as in list of word in vocab
    '''
    # TODO ======================== YOUR CODE HERE =====================================
    # The first step is to use the tokenize function to process each sentence into list of words.
    # Then you may loop through each word of each document (sentence)

    doc_freqs = Counter()

    # Loop through each sentence
    for sentence in train_sentences:
        # Tokenize the sentence into words
        words = sentence.split()

        # Update the document frequencies
        doc_freqs.update(set(words))

    # Calculate the total number of documents
    num_docs = len(train_sentences)

    # Initialize a dictionary to hold the idf values
    idf_values = {}

    # Calculate the idf value for each word in the vocab
    for word in vocab:
        idf_values[word] = math.log(num_docs / (1 + doc_freqs.get(word, 0)))

    # notice that this is a one-off pre-processing for the tf-idf feature.
    # TODO =============================================================================
    return dict()



from collections import Counter

class custom_feature_extractor(feature_extractor):
    '''
    This is a template for implementing a simple n-gram based feature extractor.
    It counts the occurrences of n-grams in a sentence.
    '''
    def __init__(self, vocab, tokenizer, n=2):
        super().__init__(vocab, tokenizer)
        self.n = n

    def feature_map(self, sentence):
        # Tokenize the sentence
        tokens = tokenize(sentence)

        # Generate n-grams from the tokens
        ngrams = [tuple(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]

        # Count occurrences of n-grams
        ngram_counts = Counter(ngrams)

        return ngram_counts

    def __call__(self, sentence):
        # Call the feature_map method when the object is called
        return self.feature_map(sentence)
