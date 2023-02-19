"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import numpy as np

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    tag_pair_dict = {}
    emission_dict = {}

    prev_word = ''
    prev_tag = ''
    for sentence in train:
        for word in sentence:
            if (word[0] == 'START'):
                prev_word = word[0] 
                prev_tag = word[1]
            elif (word[0] == 'END'):
                if prev_tag not in tag_pair_dict:
                    tag_pair_dict[prev_tag] = {}
                    tag_pair_dict[prev_tag][word[1]] = 1
                else:
                    if word[1] not in tag_pair_dict[prev_tag]:
                        tag_pair_dict[prev_tag][word[1]] = 1
                    else:
                        tag_pair_dict[prev_tag][word[1]] += 1
            else:
                if prev_tag not in tag_pair_dict:
                    tag_pair_dict[prev_tag] = {}
                    tag_pair_dict[prev_tag][word[1]] = 1
                else:
                    if word[1] not in tag_pair_dict[prev_tag]:
                        tag_pair_dict[prev_tag][word[1]] = 1
                    else:
                        tag_pair_dict[prev_tag][word[1]] += 1

                if (word[1] not in emission_dict):
                    emission_dict[word[1]] = {}
                    emission_dict[word[1]][word[0]] = 1
                else:
                    if (word[0] not in emission_dict[word[1]]):
                        emission_dict[word[1]][word[0]] = 1
                    else:
                        emission_dict[word[1]][word[0]] += 1
                prev_word = word[0] 
                prev_tag = word[1]
    
#build trellis
    #print(tag_pair_dict['PERIOD'])
    #print(tag_pair_dict['START'])
    #print(tag_pair_dict)
    
    lap_const = 0.0001
    lap_const_tag = 0.0001

    # convert count to prob
    for t in emission_dict.keys():
        V = len(emission_dict[t]) #num of unique words for tag T
        n = sum(emission_dict[t].values()) #num of all words for tag T
        for w in emission_dict[t]:
            emission_dict[t][w] = np.log((emission_dict[t][w] + lap_const) / (n + lap_const * (V + 1)))
        emission_dict[t]['UNK'] = np.log(lap_const / (n + lap_const * (V + 1)))
    
    for t in tag_pair_dict.keys():
        V = len(tag_pair_dict[t]) #num 0f unique words for tag T
        n = sum(tag_pair_dict[t].values()) #num of all words for tag T
        for t_last in tag_pair_dict[t]:
            tag_pair_dict[t][t_last] = np.log((tag_pair_dict[t][t_last] + lap_const_tag) / (n + lap_const_tag * (V + 1)))
        tag_pair_dict[t]['UNK'] = np.log(lap_const_tag / (n + lap_const_tag * (V + 1)))
    
    #print(tag_pair_dict['START'])
    #print(tag_pair_dict)

    res = []
    for sentence in test:
        trellis_v = []
        trellis_b = []
        #print(sentence)
        for i in range(len(sentence)):
        #for i in range(4):
            word_tags = {}
            word_prev = {}
            
            #first word in the sentence
            if (prev_word == 'START'):
                for k in emission_dict.keys():
                    if (k not in tag_pair_dict['START']):
                        p_s = tag_pair_dict['START']['UNK']
                    else:
                        p_s = tag_pair_dict['START'][k]
                    
                    if (sentence[1] not in emission_dict[k]):
                        p_e = emission_dict[k]['UNK']
                    else:
                        p_e = emission_dict[k][sentence[1]]
                    # print(sentence[i])
                    # print(p_e)
                    # print(p_s)
                    word_tags[k] = p_s + p_e
                    word_prev[k] = 'START'
                trellis_v.append(word_tags)
                trellis_b.append(word_prev)
                prev_word = sentence[i]     
            elif (sentence[i] == 'START'):
                prev_word = 'START'

            elif (sentence[i] == 'END'):
                sens_tag = backtrack(trellis_v, trellis_b, sentence)
                #print("sentence ",sentence)
                #print("tag ", sens_tag)
                res.append(sens_tag)
                # print(trellis_v)
                # print(trellis_b)
                # print(sens_tag)
                # 1/0

            else:
                
                for k in emission_dict.keys():
                    if (sentence[i] not in emission_dict[k]):
                        p_e = emission_dict[k]['UNK']
                    else:
                        p_e = emission_dict[k][sentence[i]]

                    prob_max = -float('inf')
                    tag_max = ''
                    for tag_key in trellis_v[i - 2]:
                        if (k not in tag_pair_dict[tag_key]):
                            p_t = tag_pair_dict[tag_key]['UNK']
                        else:
                            p_t = tag_pair_dict[tag_key][k]

                        if (trellis_v[i-2][tag_key] + p_t + p_e) > prob_max:
                            prob_max = trellis_v[i-2][tag_key] + p_t + p_e
                            tag_max = tag_key
                    word_tags[k] = prob_max
                    word_prev[k] = tag_max

                trellis_v.append(word_tags)
                trellis_b.append(word_prev)
                prev_word = sentence[i]
                
                # print(trellis_v)
                # print(trellis_b)
                # print('jkjhvhv') 
            
    return res

#t_v : list of dict, each dict store the prob of each tag - eg: t_v[i]['ADJ'] = 0.01
#t_b : list of dict, each dict store the prev tag - eg: t_b[i]['ADJ'] = 'ADV'
def backtrack(t_v, t_b, sentence):
    ret_tags = []
    ret_tags.append(('END', 'END'))
    curr_tag = max(t_v[-1], key=t_v[-1].get)
    ret_tags.insert(0, (sentence[-2], curr_tag))
    i = len(t_b)-1

    while(t_b[i][curr_tag] != 'START'):
        ret_tags.insert(0, (sentence[i], t_b[i][curr_tag]))
        curr_tag = t_b[i][curr_tag]
        i -= 1
    ret_tags.insert(0, ('START', 'START'))
    # print(ret_tags)
    # print(sentence)
    return ret_tags


