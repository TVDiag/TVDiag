import os.path

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from helper import io
# tfidf * word embedding


# adapted from https://anonymous.4open.science/r/DiagFusion-378D

def read_text(path):
    text = []
    f = open(path, 'r')
    line = f.readline()
#     text.append(line[:-12])
    text.append(line.split('\t')[0])
    while line:
        line = f.readline()
#         text.append(line[:-12])
        text.append(line.split('\t')[0])
    f.close()
    return text[:-1]


def sentence_embedding(type, source_path, train_path, test_path, save_path, service_num):
    data_dict = io.load(source_path)

    train_text = read_text(train_path)
    test_text = read_text(test_path)
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\S\S+')
    transformer = TfidfTransformer()
    vec_train = vectorizer.fit_transform(train_text)
    tfidf_train = transformer.fit_transform(vec_train)
    vec_test = vectorizer.transform(test_text)
    tfidf_test = transformer.transform(vec_test)

    weight_train = tfidf_train.toarray()
    weight_test = tfidf_test.toarray()
#     weight_test = tfidf_test.toarray()[-len(test_text): ]

    word = vectorizer.get_feature_names_out()
    word_dict = {word[i]: i for i in range(len(word))}

    
#     print('dict(vectorizer words) - dict(fasttext words) = ', set(word_dict.keys()-set(data_dict.keys())))
#     assert len(word_dict) + 1 == len(data_dict)
    print('len vectorizer words:', len(word_dict))
    print('len fasttext words:', len(data_dict))
    print('dict(fasttext words) - dict(vectorizer words) = ', set(data_dict.keys()-set(word_dict.keys())))
    print('dict(vectorizer words) - dict(fasttext words) = ', set(word_dict.keys()-set(data_dict.keys())))

    train_embedding = tfidf_word_embedding(weight_train, data_dict, train_text, word_dict, service_num)
    test_embedding = tfidf_word_embedding(weight_test, data_dict, test_text, word_dict, service_num)

    train_embedding.extend(test_embedding)

    print('sentence_embedding shape:', f'{len(train_embedding)} * {len(train_embedding[0])} * {len(train_embedding[0][0])}')
    io.save(save_path, train_embedding)


def tfidf_word_embedding(weight, data_dict, texts, word_dict, service_num):
    length = len(data_dict[list(data_dict.keys())[0]])
    count = 0
    case_embedding = []
    sentence_embedding = []
    for text in texts:
        temp = np.array([0] * length, 'float32')
        count_log = count_metric = count_trace = 0
        if text != '':
            words = list(set(text.split(' ')))
#             words = text.split()
            for word in words:
                if word in word_dict:
                    temp = temp + weight[count][word_dict[word]] * np.array(data_dict[word])
#                     temp = temp + np.array(data_dict[word])
#             temp /= len(words)
        case_embedding.append(temp)
        if (count + 1) % service_num == 0:  #@
            sentence_embedding.append(case_embedding)
            case_embedding = []
        count += 1
    return sentence_embedding


def run_sentence_embedding(type: str, data_path: str, save_path: str, node_num: int):

    source_path = os.path.join(data_path, 'fasttext/{}_event_embedding.pkl'.format(type))
    train_path = os.path.join(data_path, 'fasttext/temp', '{}_train.txt'.format(type))
    test_path = os.path.join(data_path, 'fasttext/temp', '{}_test.txt'.format(type))

    sentence_embedding(type, source_path, train_path, test_path, save_path, node_num)



