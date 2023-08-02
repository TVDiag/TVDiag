import random
import fasttext
import numpy as np
from helper import io
import time
import os

# adapted from https://anonymous.4open.science/r/DiagFusion-378D

class FastTextLab:
    def __init__(self, type, data_path, demo_datas, nodes, labels, embedding_dim):
        self.labels = labels
        self.type = type
        self.data_path = data_path

        self.method = fasttext.train_supervised
        self.embedding_dim = embedding_dim

        self.nodes = nodes
        self.demo_datas = demo_datas

        self.anomaly_types = np.append('[normal]', labels['anomaly_type'].unique())
        self.anomaly_type_labels = dict(zip(self.anomaly_types, range(len(self.anomaly_types))))
        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))
        # print(self.anomaly_type_labels)
        self.train_data, self.test_data = self.prepare_data()

    def prepare_data(self):
        train = self.labels[self.labels['data_type'] == 'train'].index
        test = self.labels[self.labels['data_type'] == 'test'].index

        self.train_path = os.path.join(self.data_path, 'fasttext/temp', '{}_train.txt'.format(self.type))
        self.test_path = os.path.join(self.data_path, 'fasttext/temp', '{}_test.txt'.format(self.type))

        self.save_to_txt(self.demo_datas, train, self.train_path)
        self.save_to_txt(self.demo_datas, test, self.test_path)

        with open(self.train_path, 'r') as f:
            train_data = f.read().splitlines()
        with open(self.test_path, 'r') as f:
            test_data = f.read().splitlines()
        return train_data, test_data

    def w2v_DA(self):
        da_train_data = self.train_data.copy()
        model = self.method(self.train_path, dim=self.embedding_dim, minCount=1, minn=0, maxn=0, epoch=5)
        for anomaly_type in self.anomaly_types:
            for node in self.nodes:
                sample_count = len([
                    text for text in self.train_data
                    if text.split('__label__')[-1] == str(self.node_labels[node]) + str(
                        self.anomaly_type_labels[anomaly_type])])
                if sample_count == 0:
                    continue
                anomaly_texts = [
                    text for text in self.train_data
                    if text.split('\t')[
                           -1] == f'__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}']
                loop = 0
                sample_num = 1000
                while sample_count < sample_num:
                    loop += 1
                    if loop >= 10 * sample_num:
                        break

                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < 1:
                        continue

                    edit_event_ids = random.sample(range(len(chosen_text_splits)), 1)
                    for event_id in edit_event_ids:

                        nearest_event = model.get_nearest_neighbors(chosen_text_splits[event_id])[0][-1]
                        chosen_text_splits[event_id] = nearest_event
                    da_train_data.append(
                        ' '.join(
                            chosen_text_splits) + f'\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}')
                    sample_count += 1

        #                 words = []
        self.train_da_path = os.path.join(self.data_path, 'fasttext/temp', '{}_train_da.txt'.format(self.type))
        with open(self.train_da_path, 'w') as f:
            for text in da_train_data:
                f.write(text + '\n')

    def event_embedding_lab(self, data_path):
        model = self.method(data_path, dim=self.embedding_dim,
                            minCount=1, minn=0, maxn=0, epoch=5)
        event_dict = dict()
        for event in model.words:
            event_dict[event] = model[event]
        return event_dict

    def save_to_txt(self, data: dict, keys, save_path):
        fillna = False
        with open(save_path, 'w') as f:
            for case_id in keys:
                case_id = case_id if case_id in data.keys() else str(case_id)
                for node_info in data[case_id]:
                    text = data[case_id][node_info]
                    if isinstance(text, str):
                        text = text.replace('(', '').replace(')', '')
                        if fillna and len(text) == 0:
                            text = 'None'
                        f.write(
                            f'{text}\t__label__{self.node_labels[node_info[0]]}{self.anomaly_type_labels[node_info[1]]}\n')
                    #                         self.anomaly_types.add(f'{node_info[0]}{node_info[1]}')
                    elif isinstance(text, list):
                        text = ' '.join(text)
                        if fillna and len(text) == 0:
                            text = 'None'
                        f.write(
                            f'{text}\t__label__{self.node_labels[node_info[0]]}{self.anomaly_type_labels[node_info[1]]}\n')
                    #                         self.anomaly_types.add(f'{node_info[0]}{node_info[1]}')
                    else:
                        raise Exception('type error')
        return

    def do_lab(self):
        self.w2v_DA()
        event_embedding_path = os.path.join(self.data_path, 'fasttext', '{}_event_embedding.pkl'.format(self.type))
        io.save(event_embedding_path, self.event_embedding_lab(self.train_da_path))


def run_fasttext(type: str, data_path, demo_datas, nodes, labels, embedding_dim):
    print('============================ process the {} ================================='.format(type))

    start_ts = time.time()
    lab2 = FastTextLab(type, data_path, demo_datas, nodes, labels, embedding_dim)
    lab2.do_lab()
    end_ts = time.time()
    print('fasttext time used:', end_ts - start_ts, 's')
