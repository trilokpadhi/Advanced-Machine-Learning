from classifier import load_data, tokenize, feature_extractor, classifier_agent, tfidf_extractor, custom_feature_extractor
from collections import Counter
from nltk.tokenize import word_tokenize     
import numpy as np
import random
from gensim.models import KeyedVectors

def load_and_process_data(pos_file, neg_file):
    sentences_pos = load_data(pos_file)
    sentences_neg = load_data(neg_file)
    sentences = sentences_pos + sentences_neg
    labels = [1]*len(sentences_pos) + [0]*len(sentences_neg)
    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    sentences, labels = zip(*combined)
    return list(sentences), list(labels)

def train_model(feat_map, train_sentences, train_labels, d, niter, lr):
    params = np.random.randn(d) * 0.01
    classifier = classifier_agent(feat_map, params)
    classifier.train_gd(train_sentences, train_labels, niter, lr)
    return classifier

def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        vocab_list = [item.strip() for item in file.readlines()]

    print("Loading and processing data ...")
    train_sentences, train_labels = load_and_process_data("data/training_pos.txt", "data/training_neg.txt")
    test_sentences, test_labels = load_and_process_data("data/test_pos_public.txt", "data/test_neg_public.txt")

    print("Creating bag of words feature extractor ...")
    feat_map = feature_extractor(vocab_list, tokenize)

    print("Training using Bag of words and GD for 500 iterations.")
    classifier1 = train_model(feat_map, train_sentences, train_labels, len(vocab_list), 10, 0.02)

    print("Training using Bag of words and SGD for 10 data passes.")
    classifier2 = train_model(feat_map, train_sentences, train_labels, len(vocab_list), 10, 0.02)

    print("Creating Tfidf feature extractor ...")
    all_text = ' '.join(train_sentences)
    all_words = word_tokenize(all_text)
    word_freq = Counter(all_words)
    feat_map_extractor = tfidf_extractor(vocab_list, tokenize, word_freq)

    print("Training using Tfidf and GD for 10 iterations.")
    classifier3 = train_model(feat_map_extractor.tfidf_feature, train_sentences, train_labels, len(vocab_list), 10, 0.02)

    print("Creating custom feature extractor ...")
    custom_feat_map_extractor = custom_feature_extractor(vocab_list, tokenize, word_freq)

    print("Training using custom features and GD for 10 iterations.")
    classifier4 = train_model(feat_map_extractor.tfidf_feature, train_sentences, train_labels, len(vocab_list), 10, 0.02)

    err1 = classifier1.eval_model(test_sentences,test_labels)
    err2 = classifier2.eval_model(test_sentences,test_labels)
    err3 = classifier3.eval_model(test_sentences,test_labels)
    err4 = classifier4.eval_model(test_sentences,test_labels)

    print('Bag of words + GD: test err = ', err1,
          'Bag of words + SGD: test err = ', err2,
          'Tfidf + GD: test err = ', err3,
          'Custom features + GD: test err = ', err4)
    
if __name__ == "__main__":
    main()