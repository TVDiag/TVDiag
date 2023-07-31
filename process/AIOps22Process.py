import math

from torch.utils.data import DataLoader
from tqdm import tqdm
from process.events import fasttext_with_DA, sententce_embedding
from process.BaseProcess import Process
from core.multimodal_dataset import MultiModalDataSet
from helper.paths import *
from helper import io
import json
import pandas as pd
import numpy as np
import os


class AIOps22Process(Process):

    def __init__(self, config: dict):
        super().__init__(config)
        self.trace_embedding_path = None
        self.metric_embedding_path = None
        self.data_path = None
        self.metric_embedding_dim = None
        self.trace_embedding_dim = None

    def process(self, reconstruct=False):
        self.metric_embedding_dim = self.dataset_args['metric_embedding_dim']
        self.trace_embedding_dim = self.dataset_args['trace_embedding_dim']
        self.log_embedding_dim = self.dataset_args['log_embedding_dim']
        self.data_path = os.path.join(dataset_path, self.dataset_name)

        self.metric_embedding_path = os.path.join(self.data_path, self.dataset_args['sentence_embedding']['metric_embedding_path'])
        self.trace_embedding_path = os.path.join(self.data_path, self.dataset_args['sentence_embedding']['trace_embedding_path'])
        self.log_embedding_path = os.path.join(self.data_path, self.dataset_args['sentence_embedding']['log_embedding_path'])

        label_path = os.path.join(self.data_path, self.dataset_args['labels'])
        metric_path = os.path.join(self.data_path, self.dataset_args['metrics'])
        trace_path = os.path.join(self.data_path, self.dataset_args['traces'])
        log_path = os.path.join(self.data_path, self.dataset_args['logs'])
        edge_path = os.path.join(self.data_path, self.dataset_args['edges'])
        node_path = os.path.join(self.data_path, self.dataset_args['nodes'])

        self.labels = pd.read_csv(label_path)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp)
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)

        self.edges = io.load(edge_path)
        self.nodes = io.load(node_path)

        self.labels.rename(columns={'cmdb_id': 'instance', 'failure_type': 'anomaly_type'}, inplace=True)
        self.labels.set_index('index', inplace=True)

        if reconstruct:
            self.build_embedding()
        return self.build_dataset()

    def build_embedding(self):
        for k, v in self.metrics.items():
            self.metrics[k] = [x for x in v if not math.isinf(x[3])]

        anomaly_instance = list(self.labels['instance'])
        anomaly_type = list(self.labels['anomaly_type'])

        k = 0
        demo_metrics = {x: {} for x in self.labels.index}
        demo_traces = {x: {} for x in self.labels.index}
        demo_logs = {x: {} for x in self.labels.index}

        for case_id, v in tqdm(demo_traces.items()):
            anomaly_instance_name = anomaly_instance[k]
            anomaly_service_type = anomaly_type[k]
            k += 1
            inner_dict_key = [(x, anomaly_service_type) if x == anomaly_instance_name else (x, "[normal]") for x in
                              self.nodes]
            # metric
            if not self.metrics is None:
                demo_metrics[case_id] = {
                    x: [[int(y[0]), "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")] for y in
                        self.metrics[str(case_id)]
                        if
                        x[0] in y[1]] for x in inner_dict_key}
            # trace
            if not self.traces is None:
                demo_traces[case_id] = {
                    x: [[int(y[0]), "{}_{}".format(y[1], y[2])] for y in self.traces[str(case_id)] if x[0] in y[1] or x[0] in y[2]] for x in inner_dict_key}
                # for inner_key in inner_dict_key:
                #     demo_metrics[case_id][inner_key].extend(
                #         [[y[0], "{}_{}".format(y[1], y[2])] for y in self.traces[str(case_id)]
                #          if y[1] == inner_key[0] or y[2] == inner_key[0]])

            # log
            if not self.logs is None:
                demo_logs[case_id] = {
                    x: [[int(y[0]), y[2]] for y in self.logs[case_id] if
                        y[1] == x[0]] for x in inner_dict_key}
                # for inner_key in inner_dict_key:
                #     demo_metrics[case_id][inner_key].extend([[y[0], y[2]] for y in self.logs[case_id] if y[1] == inner_key[0]])

            for inner_key in inner_dict_key:
                temp = demo_metrics[case_id][inner_key]
                sort_list = sorted(temp, key=lambda x: x[0])
                temp_list = [x[1] for x in sort_list]
                demo_metrics[case_id][inner_key] = ' '.join(temp_list)

                temp = demo_traces[case_id][inner_key]
                sort_list = sorted(temp, key=lambda x: x[0])
                temp_list = [x[1] for x in sort_list]
                demo_traces[case_id][inner_key] = ' '.join(temp_list)

                temp = demo_logs[case_id][inner_key]
                sort_list = sorted(temp, key=lambda x: x[0])
                temp_list = [x[1] for x in sort_list]
                demo_logs[case_id][inner_key] = ' '.join(temp_list)

        # io.save(os.path.join(self.data_path, 'log_events.pkl'), demo_logs)
        # io.save(os.path.join(self.data_path, 'metric_events.pkl'), demo_metrics)
        # io.save(os.path.join(self.data_path, 'trace_events.pkl'), demo_traces)

        fasttext_with_DA.run_fasttext('metric', self.data_path, demo_metrics, self.nodes, self.labels, self.metric_embedding_dim)
        fasttext_with_DA.run_fasttext('trace', self.data_path, demo_traces, self.nodes, self.labels, self.trace_embedding_dim)
        fasttext_with_DA.run_fasttext('log', self.data_path, demo_logs, self.nodes, self.labels, self.log_embedding_dim)

        sententce_embedding.run_sentence_embedding('metric', self.data_path, self.metric_embedding_path, len(self.nodes))
        sententce_embedding.run_sentence_embedding('trace', self.data_path, self.trace_embedding_path, len(self.nodes))
        sententce_embedding.run_sentence_embedding('log', self.data_path, self.log_embedding_path, len(self.nodes))

    def build_dataset(self):
        metrics_embedding = io.load(self.metric_embedding_path)
        traces_embedding = io.load(self.trace_embedding_path)
        logs_embedding = io.load(self.log_embedding_path)

        metric_Xs = np.array(metrics_embedding)
        trace_Xs = np.array(traces_embedding)
        log_Xs = np.array(logs_embedding)

        label_types = ['anomaly_type', 'instance']
        label_dict = {label_type: None for label_type in label_types}
        for label_type in label_types:
            label_dict[label_type] = self.get_label(label_type, self.labels)

        train_index = np.where(self.labels['data_type'].values == 'train')
        test_index = np.where(self.labels['data_type'].values == 'test')

        train_metric_Xs = metric_Xs[train_index]
        train_trace_Xs = trace_Xs[train_index]
        train_log_Xs = log_Xs[train_index]
        # train_service_labels = label_dict['service'][train_index]
        train_instance_labels = label_dict['instance'][train_index]
        train_type_labels = label_dict['anomaly_type'][train_index]

        test_metric_Xs = metric_Xs[test_index]
        test_trace_Xs = trace_Xs[test_index]
        test_log_Xs = log_Xs[test_index]
        # test_service_labels = label_dict['service'][test_index]
        test_instance_labels = label_dict['instance'][test_index]
        test_type_labels = label_dict['anomaly_type'][test_index]

        train_data = MultiModalDataSet(train_metric_Xs, train_trace_Xs, train_log_Xs, train_instance_labels,train_type_labels, self.nodes, self.edges)
        test_data = MultiModalDataSet(test_metric_Xs, test_trace_Xs, test_log_Xs, test_instance_labels, test_type_labels, self.nodes, self.edges)

        return train_data, test_data

    def get_label(self, label_type, run_table):
        meta_labels = sorted(list(set(list(run_table[label_type]))))
        labels_idx = {label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))}
        labels = np.array(run_table[label_type].apply(lambda label_str: labels_idx[label_str]))
        return labels