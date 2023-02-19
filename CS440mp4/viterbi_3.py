"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""
import numpy as np
def viterbi_3(train, test):
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
    hapax_const = 0.00001
    hapax = {}
    hapax_ed = {}
    hapax_ing = {}
    hapax_ity = {}
    hapax_s = {}
    hapax_ly = {}
    hapax_num = {}
    hapax_cs = {}
    hapax_tion = {}
    hapax_er = {}
    hapax_ment = {}
    hapax_tic = {}
    hapax_able = {}
    hapax_d = {}
    hapax_ship = {}
    hapax_dash = {}
    hapax_al = {}

    total_w = 0
    total_w_ed = 0
    total_w_ing = 0
    total_w_ity = 0
    total_w_s = 0
    total_w_ly = 0
    total_w_num = 0
    total_w_cs = 0
    total_w_tion = 0
    total_w_er = 0
    total_w_ment = 0
    total_w_tic = 0
    total_w_able = 0
    total_w_d = 0
    total_w_ship = 0
    total_w_dash = 0
    total_w_al = 0

    temp = []
    for t in emission_dict.keys():
        hapax_ed[t] = 0
        hapax_ing[t] = 0
        hapax_ity[t] = 0
        hapax_s[t] = 0
        hapax_ly[t] = 0
        hapax_num[t] = 0
        hapax_cs[t] = 0
        hapax_tion[t] = 0
        hapax_er[t] = 0
        hapax_ment[t] = 0
        hapax_tic[t] = 0
        hapax_able[t] = 0
        hapax_d[t] = 0
        hapax_ship[t] = 0
        hapax_dash[t] = 0
        hapax_al[t] = 0
        for w in emission_dict[t]:
            if (emission_dict[t][w] == 1):
                if (w[-2:] != "ed") and (w[-2:] != "ly") and (w[-2:] != "er") and (w[-2:] != "'s") and (w[-2:] != "al"):
                    if (w[-3:] != "ing") and (w[-3:] !="ity") and (w[-3:] !="tic"):
                        if (w[-4:] != "tion") and (w[-4:] !="ment") and (w[-4:] !="able") and (w[-4:] !="ship"):
                            if (w[-1] != 's') and not(w.isnumeric()):

                                temp.append(w)
                if (w[-2:] == "ed"):
                    hapax_ed[t] += 1
                    total_w_ed += 1
                if (w[-3:] == "ing"):
                    hapax_ing[t] += 1
                    total_w_ing += 1
                if (w[-3:] == "ity"):
                    hapax_ity[t] += 1
                    total_w_ity += 1
                if (w[-1] == "s"):
                    hapax_s[t] += 1
                    total_w_s += 1
                if (w[-2:] == "ly"):
                    hapax_ly[t] += 1
                    total_w_ly += 1
                if (w.isnumeric()):
                    hapax_num[t] += 1
                    total_w_num += 1
                if (w[-2:] == "'s") or (w[-2:] == "s'"):
                    hapax_cs[t] += 1
                    total_w_cs += 1
                if (w[-4:] == "tion"):
                    hapax_tion[t] += 1
                    total_w_tion += 1
                if (w[-2:] == "er"):
                    hapax_er[t] += 1
                    total_w_er += 1
                if (w[-4:] == "ment"):
                    hapax_ment[t] += 1
                    total_w_ment += 1
                if (w[-3:] == "tic"):
                    hapax_tic[t] += 1
                    total_w_tic += 1
                if (w[-4:] == "able"):
                    hapax_able[t] += 1
                    total_w_able += 1
                if (w[0] == "$"):
                    hapax_d[t] += 1
                    total_w_d += 1
                if (w[-4:] == "ship"):
                    hapax_ship[t] += 1
                    total_w_ship += 1
                if (has_dash(w)):
                    hapax_dash[t] += 1
                    total_w_dash += 1
                if (w[-2:] == "al"):
                    hapax_al[t] += 1
                    total_w_al += 1
                total_w += 1
                if t not in hapax:
                    hapax[t] = 1
                else:
                    hapax[t] += 1
    # print("hapax_ed", sum(hapax_ed.values()), hapax_ed)
    # print("hapax_ing", sum(hapax_ing.values()), hapax_ing)
    # print("hapax_s", sum(hapax_s.values()), hapax_s)
    # 1/0
    #print(sum(hapax.values()))
    #print(temp)
    for t in emission_dict.keys():
        if t not in hapax.keys():
            hapax[t] = hapax_const / (total_w + hapax_const * (len(emission_dict) + 1))
        else: 
            hapax[t] = (hapax[t] + hapax_const) / (total_w + hapax_const * (len(emission_dict) + 1))

    hapax_const_ed = 0.1
    hapax_const_ing = 0.1
    hapax_const_ity = 0.075
    hapax_const_s = 0.05
    hapax_const_ly = 0.08
    hapax_const_num = 0.005
    hapax_const_cs = 0.05
    hapax_const_tion = 0.01
    hapax_const_er = 0.005
    hapax_const_ment = 0.01
    hapax_const_tic = 0.01
    hapax_const_able = 0.01
    hapax_const_al = 0.08

    for t in emission_dict.keys():
        if hapax_ed[t] == 0:
            hapax_ed[t] = hapax_const_ed / (total_w_ed + hapax_const_ed * (len(emission_dict) + 1))
        else: 
            hapax_ed[t] = (hapax_ed[t] + hapax_const_ed) / (total_w_ed + hapax_const_ed * (len(emission_dict) + 1))

        if hapax_ing[t] == 0:
            hapax_ing[t] = hapax_const_ing / (total_w_ing + hapax_const_ing * (len(emission_dict) + 1))
        else: 
            hapax_ing[t] = (hapax_ing[t] + hapax_const_ing) / (total_w_ing + hapax_const_ing * (len(emission_dict) + 1))
        
        if hapax_ity[t] == 0:
            hapax_ity[t] = hapax_const_ity / (total_w_ity + hapax_const_ity * (len(emission_dict) + 1))
        else: 
            hapax_ity[t] = (hapax_ity[t] + hapax_const_ity) / (total_w_ity + hapax_const_ity * (len(emission_dict) + 1))

        if hapax_s[t] == 0:
            hapax_s[t] = hapax_const_s / (total_w_s + hapax_const_s * (len(emission_dict) + 1))
        else: 
            hapax_s[t] = (hapax_s[t] + hapax_const_s) / (total_w_s + hapax_const_s * (len(emission_dict) + 1))

        if hapax_ly[t] == 0:
            hapax_ly[t] = hapax_const_ly / (total_w_ly + hapax_const_ly * (len(emission_dict) + 1))
        else: 
            hapax_ly[t] = (hapax_ly[t] + hapax_const_ly) / (total_w_ly + hapax_const_ly * (len(emission_dict) + 1))
        
        if hapax_num[t] == 0:
            hapax_num[t] = hapax_const_num / (total_w_num + hapax_const_num * (len(emission_dict) + 1))
        else: 
            hapax_num[t] = (hapax_num[t] + hapax_const_num) / (total_w_num + hapax_const_num * (len(emission_dict) + 1))
        
        if hapax_cs[t] == 0:
            hapax_cs[t] = hapax_const_cs / (total_w_cs + hapax_const_cs * (len(emission_dict) + 1))
        else: 
            hapax_cs[t] = (hapax_cs[t] + hapax_const_cs) / (total_w_cs + hapax_const_cs * (len(emission_dict) + 1))

        if hapax_tion[t] == 0:
            hapax_tion[t] = hapax_const_tion / (total_w_tion + hapax_const_tion * (len(emission_dict) + 1))
        else: 
            hapax_tion[t] = (hapax_tion[t] + hapax_const_tion) / (total_w_tion + hapax_const_tion * (len(emission_dict) + 1))
        
        if hapax_er[t] == 0:
            hapax_er[t] = hapax_const_er / (total_w_er + hapax_const_er * (len(emission_dict) + 1))
        else: 
            hapax_er[t] = (hapax_er[t] + hapax_const_er) / (total_w_er + hapax_const_er * (len(emission_dict) + 1))
        
        if hapax_ment[t] == 0:
            hapax_ment[t] = hapax_const_ment / (total_w_ment + hapax_const_ment * (len(emission_dict) + 1))
        else: 
            hapax_ment[t] = (hapax_ment[t] + hapax_const_ment) / (total_w_ment + hapax_const_ment * (len(emission_dict) + 1))
        
        if hapax_tic[t] == 0:
            hapax_tic[t] = hapax_const_tic / (total_w_tic + hapax_const_tic * (len(emission_dict) + 1))
        else: 
            hapax_tic[t] = (hapax_tic[t] + hapax_const_tic) / (total_w_tic + hapax_const_tic * (len(emission_dict) + 1))

        if hapax_able[t] == 0:
            hapax_able[t] = hapax_const_able / (total_w_able + hapax_const_able * (len(emission_dict) + 1))
        else: 
            hapax_able[t] = (hapax_able[t] + hapax_const_able) / (total_w_able + hapax_const_able * (len(emission_dict) + 1))
        
        if hapax_d[t] == 0:
            hapax_d[t] = hapax_const_s / (total_w_d + hapax_const_s * (len(emission_dict) + 1))
        else: 
            hapax_d[t] = (hapax_d[t] + hapax_const_s) / (total_w_d + hapax_const_s * (len(emission_dict) + 1))

        if hapax_ship[t] == 0:
            hapax_ship[t] = hapax_const_s / (total_w_ship + hapax_const_s * (len(emission_dict) + 1))
        else: 
            hapax_ship[t] = (hapax_ship[t] + hapax_const_s) / (total_w_ship + hapax_const_s * (len(emission_dict) + 1))

        if hapax_dash[t] == 0:
            hapax_dash[t] = hapax_const_s / (total_w_dash + hapax_const_s * (len(emission_dict) + 1))
        else: 
            hapax_dash[t] = (hapax_dash[t] + hapax_const_s) / (total_w_dash + hapax_const_s * (len(emission_dict) + 1))
        
        if hapax_al[t] == 0:
            hapax_al[t] = hapax_const_al / (total_w_al + hapax_const_al * (len(emission_dict) + 1))
        else: 
            hapax_al[t] = (hapax_al[t] + hapax_const_al) / (total_w_al + hapax_const_al * (len(emission_dict) + 1))




    lap_const = 0.005
    lap_const_tag = 0.001

    # convert count to prob
    for t in emission_dict.keys():
        V = len(emission_dict[t]) #num of unique words for tag T
        n = sum(emission_dict[t].values()) #num of all words for tag T
            
        for w in emission_dict[t]:

            emission_dict[t][w] = np.log((emission_dict[t][w] + lap_const * hapax[t]) / (n + hapax[t] * lap_const * (V + 1)))

        emission_dict[t]['UNK'] = np.log((lap_const * hapax[t]) / (n + lap_const * hapax[t] * (V + 1)))
        emission_dict[t]['S-ED'] = np.log((lap_const * hapax[t] * hapax_ed[t]) / (n + lap_const * hapax[t] * hapax_ed[t] * (V + 1)))
        emission_dict[t]['S-ING'] = np.log((lap_const * hapax[t] * hapax_ing[t]) / (n + lap_const * hapax[t] * hapax_ing[t] * (V + 1)))
        emission_dict[t]['S-ITY'] = np.log((lap_const * hapax[t] * hapax_ity[t]) / (n + lap_const * hapax[t] * hapax_ity[t] * (V + 1)))
        emission_dict[t]['S-S'] = np.log((lap_const * hapax[t] * hapax_s[t]) / (n + lap_const * hapax[t] * hapax_s[t] * (V + 1)))
        emission_dict[t]['S-LY'] = np.log((lap_const * hapax[t] * hapax_ly[t]) / (n + lap_const * hapax[t] * hapax_ly[t] * (V + 1)))
        emission_dict[t]['S-NUM'] = np.log((lap_const * hapax[t] * hapax_num[t]) / (n + lap_const * hapax[t] * hapax_num[t] * (V + 1)))
        emission_dict[t]['S-CS'] = np.log((lap_const * hapax[t] * hapax_cs[t]) / (n + lap_const * hapax[t] * hapax_cs[t] * (V + 1)))
        emission_dict[t]['S-TION'] = np.log((lap_const * hapax[t] * hapax_tion[t]) / (n + lap_const * hapax[t] * hapax_tion[t] * (V + 1)))
        emission_dict[t]['S-ER'] = np.log((lap_const * hapax[t] * hapax_er[t]) / (n + lap_const * hapax[t] * hapax_er[t] * (V + 1)))
        emission_dict[t]['S-MENT'] = np.log((lap_const * hapax[t] * hapax_ment[t]) / (n + lap_const * hapax[t] * hapax_ment[t] * (V + 1)))
        emission_dict[t]['S-TIC'] = np.log((lap_const * hapax[t] * hapax_tic[t]) / (n + lap_const * hapax[t] * hapax_tic[t] * (V + 1)))
        emission_dict[t]['S-ABLE'] = np.log((lap_const * hapax[t] * hapax_able[t]) / (n + lap_const * hapax[t] * hapax_able[t] * (V + 1)))
        emission_dict[t]['S-D'] = np.log((lap_const * hapax[t] * hapax_d[t]) / (n + lap_const * hapax[t] * hapax_d[t] * (V + 1)))
        emission_dict[t]['S-SHIP'] = np.log((lap_const * hapax[t] * hapax_ship[t]) / (n + lap_const * hapax[t] * hapax_ship[t] * (V + 1)))
        emission_dict[t]['S-DASH'] = np.log((lap_const * hapax[t] * hapax_dash[t]) / (n + lap_const * hapax[t] * hapax_dash[t] * (V + 1)))
        emission_dict[t]['S-AL'] = np.log((lap_const * hapax[t] * hapax_al[t]) / (n + lap_const * hapax[t] * hapax_al[t] * (V + 1)))
    
    
    for t in tag_pair_dict.keys():
        V = len(tag_pair_dict[t]) #num 0f unique words for tag T
        n = sum(tag_pair_dict[t].values()) #num of all words for tag T
        for t_last in tag_pair_dict[t]:
            
            tag_pair_dict[t][t_last] = np.log((tag_pair_dict[t][t_last] + lap_const_tag) / (n + lap_const_tag * (V + 1)))

        tag_pair_dict[t]['UNK'] = np.log((lap_const_tag)/ (n + lap_const_tag * (V + 1)))
    
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
                        if (sentence[i][-2:] == 'ed'):
                            p_e = emission_dict[k]['S-ED']
                        elif(sentence[i][-3:] == 'ing'):
                            p_e = emission_dict[k]['S-ING']
                        elif(sentence[i][-3:] == 'ity'):
                            p_e = emission_dict[k]['S-ITY']
                        elif(sentence[i][-2:] == "'s") or (sentence[i][-2:] == "s'"):
                            #print(sentence[i])
                            p_e = emission_dict[k]['S-CS']    
                        elif(sentence[i][-1] == 's'):
                            p_e = emission_dict[k]['S-S']
                        elif(sentence[i][-2:] == 'ly'):
                            p_e = emission_dict[k]['S-LY']
                        elif(sentence[i].isnumeric()):
                            p_e = emission_dict[k]['S-NUM'] 
                        elif(sentence[i][-4:] == 'tion'):
                            p_e = emission_dict[k]['S-TION']
                        elif(sentence[i][-2:] == 'er'):
                            p_e = emission_dict[k]['S-ER']
                        elif(sentence[i][-4:] == 'ment'):
                            p_e = emission_dict[k]['S-MENT']
                        elif(sentence[i][-3:] == 'tic'):
                            p_e = emission_dict[k]['S-TIC']
                        elif(sentence[i][-4:] == 'able'):
                            p_e = emission_dict[k]['S-ABLE']
                        elif(sentence[i][0] == '$'):
                            p_e = emission_dict[k]['S-D']
                        elif(sentence[i][-4:] == 'ship'):
                            p_e = emission_dict[k]['S-SHIP']
                        elif(has_dash(sentence[i])):
                            p_e = emission_dict[k]['S-DASH']
                        elif (sentence[i][-2:] == 'al'):
                            p_e = emission_dict[k]['S-AL']
                        else:
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
                        if (sentence[i][-2:] == 'ed'):
                            p_e = emission_dict[k]['S-ED']
                        elif(sentence[i][-3:] == 'ing'):
                            p_e = emission_dict[k]['S-ING']
                        elif(sentence[i][-3:] == 'ity'):
                            p_e = emission_dict[k]['S-ITY']
                        elif(sentence[i][-2:] == "'s") or (sentence[i][-2:] == "s'"):
                            #print(sentence[i])
                            p_e = emission_dict[k]['S-CS']
                        elif(sentence[i][-1] == 's'):
                            p_e = emission_dict[k]['S-S']
                        elif(sentence[i][-2:] == 'ly'):
                            p_e = emission_dict[k]['S-LY']
                        elif(sentence[i].isnumeric()):
                            p_e = emission_dict[k]['S-NUM']
                        elif(sentence[i][-4:] == 'tion'):
                            p_e = emission_dict[k]['S-TION']
                        elif(sentence[i][-2:] == 'er'):
                            p_e = emission_dict[k]['S-ER']
                        elif(sentence[i][-4:] == 'ment'):
                            p_e = emission_dict[k]['S-MENT']
                        elif(sentence[i][-3:] == 'tic'):
                            #print('tic')
                            p_e = emission_dict[k]['S-TIC']
                        elif(sentence[i][-4:] == 'able'):
                            p_e = emission_dict[k]['S-ABLE']
                        elif(sentence[i][0] == '$'):
                            p_e = emission_dict[k]['S-D']
                            #print(p_e)
                        elif(sentence[i][-4:] == 'ship'):
                            p_e = emission_dict[k]['S-SHIP']
                        elif(has_dash(sentence[i])):
                            p_e = emission_dict[k]['S-DASH']
                        elif (sentence[i][-2:] == 'al'):
                            p_e = emission_dict[k]['S-AL']
                        else:
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

def has_dash(word):
    for char in word:
        if char == '-':
            return True
    return False

