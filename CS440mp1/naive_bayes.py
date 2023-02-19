# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader
import numpy

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.01, pos_prior=0.8,silently=False):
    pos_word_count_dict = {}
    neg_word_count_dict = {}
    total_pos = 0
    total_neg = 0

    # 6000 positive words
    # 2000 negative words
    #count the word appearance in each review class
    for i in range(len(train_set)):
        if train_labels[i] == 1: #if the word is positive
            for j in range(len(train_set[i])):
                if train_set[i][j] in pos_word_count_dict: #if the word exist in dict
                    pos_word_count_dict[train_set[i][j]] += 1
                else:
                    pos_word_count_dict[train_set[i][j]] = 1
                total_pos += 1
        else: #if the word is negative
            for k in range(len(train_set[i])):
                if train_set[i][k] in neg_word_count_dict: #if the word exist in dict
                    neg_word_count_dict[train_set[i][k]] += 1
                else:
                    neg_word_count_dict[train_set[i][k]] = 1
                total_neg += 1

    #Laplace Smooth
    V_p = len(pos_word_count_dict) #positive word types
    V_n = len(neg_word_count_dict) #negative word types
    pos_word_freq = {} #dict record P(W|positive) after smoothing
    neg_word_freq = {} #dict record P(W|negative) after smoothing

    for word in pos_word_count_dict.keys():
        pos_word_freq[word] = (pos_word_count_dict[word] + laplace) / (total_pos + laplace * (V_p + 1))

    for word in neg_word_count_dict.keys():
        neg_word_freq[word] = (neg_word_count_dict[word] + laplace) / (total_neg + laplace * (V_n + 1))

    #calculate prob of each types for each reviews
    pos_unk = laplace / ( total_pos + laplace * (len(pos_word_count_dict) + 1)) #P(UNK | positive)
    neg_unk = laplace / ( total_neg + laplace * (len(neg_word_count_dict) + 1)) #P(UNK | negative)
    yhats = []

    for i in range(len(dev_set)):
        sum_log_p_prob = 0 #log(P( all_W | positive))
        sum_log_n_prob = 0  #log(P( all_W | negative))
        for j in dev_set[i]: #loop through all words in each review
            if j in pos_word_freq:
                sum_log_p_prob += numpy.log2(pos_word_freq[j])
            else:
                sum_log_p_prob += numpy.log2(pos_unk)

        for k in dev_set[i]:
            if k in neg_word_freq:
                sum_log_n_prob += numpy.log2(neg_word_freq[k])
            else:
                sum_log_n_prob += numpy.log2(neg_unk)

        if (sum_log_p_prob + numpy.log2(pos_prior)) > (sum_log_n_prob + numpy.log2(1 - pos_prior)):
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats





def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.1, bigram_laplace=0.0001, bigram_lambda=0.15,pos_prior=0.8, silently=False):
    pos_word_count_dict = {}
    neg_word_count_dict = {}
    total_pos = 0
    total_neg = 0

    # count the word appearance in each review class
    for i in range(len(train_set)):
        if train_labels[i] == 1:  # if the word is positive
            for j in range(len(train_set[i])):
                if train_set[i][j] in pos_word_count_dict:  # if the word exist in dict
                    pos_word_count_dict[train_set[i][j]] += 1
                else:
                    pos_word_count_dict[train_set[i][j]] = 1
                total_pos += 1
        else:  # if the word is negative
            for k in range(len(train_set[i])):
                if train_set[i][k] in neg_word_count_dict:  # if the word exist in dict
                    neg_word_count_dict[train_set[i][k]] += 1
                else:
                    neg_word_count_dict[train_set[i][k]] = 1
                total_neg += 1

    pos_word_pair_count_dict = {}
    neg_word_pair_count_dict = {}
    total_pos_pair = 0
    total_neg_pair = 0

    # count the word pair appearance in each review class
    for i in range(len(train_set)):
        if train_labels[i] == 1:  # if the word is positive
            for j in range(len(train_set[i])//2):
                pos_word_pair = train_set[i][2 * j] + " " + train_set[i][2 * j + 1]
                if pos_word_pair in pos_word_pair_count_dict:  # if the word exist in dict
                    pos_word_pair_count_dict[pos_word_pair] += 1
                else:
                    pos_word_pair_count_dict[pos_word_pair] = 1
                total_pos_pair += 1
        else:  # if the word is negative
            for k in range(len(train_set[i])//2):
                neg_word_pair = train_set[i][2 * k] + " " + train_set[i][2 * k + 1]
                if neg_word_pair in neg_word_pair_count_dict:  # if the word exist in dict
                    neg_word_pair_count_dict[neg_word_pair] += 1
                else:
                    neg_word_pair_count_dict[neg_word_pair] = 1
                total_neg_pair += 1

    # Unigram Laplace Smooth
    V_p = len(pos_word_count_dict)  # positive word types
    V_n = len(neg_word_count_dict)  # negative word types
    pos_word_freq = {}  # dict record P(W|positive) after smoothing
    neg_word_freq = {}  # dict record P(W|negative) after smoothing

    for word in pos_word_count_dict.keys():
        pos_word_freq[word] = (pos_word_count_dict[word] + unigram_laplace) / (total_pos + unigram_laplace * (V_p + 1))

    for word in neg_word_count_dict.keys():
        neg_word_freq[word] = (neg_word_count_dict[word] + unigram_laplace) / (total_neg + unigram_laplace * (V_n + 1))

    # calculate unk prob of each types for each reviews
    pos_unk = unigram_laplace / (total_pos + unigram_laplace * (len(pos_word_count_dict) + 1))  # P(UNK | positive)
    neg_unk = unigram_laplace / (total_neg + unigram_laplace * (len(neg_word_count_dict) + 1))  # P(UNK | negative)

    # Bigram Laplace Smooth
    V_p_pair = len(pos_word_pair_count_dict)  # positive word types
    V_n_pair = len(neg_word_pair_count_dict)  # negative word types
    pos_word_pair_freq = {}  # dict record P(W|positive) after smoothing
    neg_word_pair_freq = {}  # dict record P(W|negative) after smoothing

    for word_pair in pos_word_pair_count_dict.keys():
        pos_word_pair_freq[word_pair] = (pos_word_pair_count_dict[word_pair] + bigram_laplace) / (total_pos_pair + bigram_laplace * (V_p_pair + 1))

    for word_pair in neg_word_pair_count_dict.keys():
        neg_word_pair_freq[word_pair] = (neg_word_pair_count_dict[word_pair] + bigram_laplace) / (total_neg_pair + bigram_laplace * (V_n_pair + 1))

    # calculate prob of each types for each reviews
    pos_pair_unk = bigram_laplace / (total_pos_pair + bigram_laplace * (len(pos_word_pair_count_dict) + 1))  # P(UNK | positive)
    neg_pair_unk = bigram_laplace / (total_neg_pair + bigram_laplace * (len(neg_word_pair_count_dict) + 1))  # P(UNK | positive)

    yhats = []

    for i in range(len(dev_set)):
        sum_log_p_prob = 0  # log(P( all_W | positive))
        sum_log_n_prob = 0  # log(P( all_W | negative))

        for j in dev_set[i]:  # loop through all words in each review
            if j in pos_word_freq:
                sum_log_p_prob += numpy.log2(pos_word_freq[j])
            else:
                sum_log_p_prob += numpy.log2(pos_unk)

        for k in dev_set[i]:
            if k in neg_word_freq:
                sum_log_n_prob += numpy.log2(neg_word_freq[k])
            else:
                sum_log_n_prob += numpy.log2(neg_unk)

        sum_log_p_pair_prob = 0 # log(P( all_W_pairs | positive))
        sum_log_n_pair_prob = 0 # log(P( all_W_pairs | negative))

        total_word_pair = len(dev_set[i])//2
        for j in range(total_word_pair):  # loop through all words pair in each review
            curr_word_pair = dev_set[i][2*j] + " " + dev_set[i][2*j + 1]
            if curr_word_pair in pos_word_pair_freq:
                sum_log_p_pair_prob += numpy.log2(pos_word_pair_freq[curr_word_pair])
            else:
                sum_log_p_pair_prob += numpy.log2(pos_pair_unk)

        for k in range(total_word_pair):
            curr_word_pair = dev_set[i][2 * k] + " " + dev_set[i][2 * k + 1]
            if curr_word_pair in neg_word_pair_freq:
                sum_log_n_pair_prob += numpy.log2(neg_word_pair_freq[curr_word_pair])
            else:
                sum_log_n_pair_prob += numpy.log2(neg_pair_unk)

        p_review_pos = (1 - bigram_lambda) * (sum_log_p_prob + numpy.log2(pos_prior)) + bigram_lambda * (sum_log_p_pair_prob + numpy.log2(pos_prior))
        p_review_neg = (1 - bigram_lambda) * (sum_log_n_prob + numpy.log2(1 - pos_prior)) + bigram_lambda * (sum_log_n_pair_prob + numpy.log2(1 - pos_prior))

        if  p_review_pos > p_review_neg :
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats

