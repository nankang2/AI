"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_dict = {}
    word_dict = {}

    for sentence in train:
        for word in sentence:
            if (word[0] == 'START') or (word[0] == 'END'):
                continue
            else:
                if word[0] not in word_dict:
                    word_dict[word[0]] = {}
                    word_dict[word[0]][word[1]] = 1
                else:
                    if word[1] not in word_dict[word[0]]:
                        word_dict[word[0]][word[1]] = 0
                    word_dict[word[0]][word[1]] += 1
                
                if word[1] not in tag_dict:
                    tag_dict[word[1]] = 0
                tag_dict[word[1]] += 1

    
    max_tag = max(tag_dict, key=tag_dict.get)

    res = []
    for sentence in test:
        res_sentence = []
        for word in sentence:
            if word in word_dict:
                #if has duplicates, use the tag has lower idx
                tag_ = max(word_dict[word], key=word_dict[word].get)
                res_sentence.append((word, tag_))
            else:
                res_sentence.append((word, max_tag))
        res.append(res_sentence)   
    
    return res