import os
from utils import io_util
from utils.time_util import *
import pandas as pd
import time
import multiprocessing as mp

def extract_df(df: pd.DataFrame, normal_peroids):
    con=None
    for p in normal_peroids:
        con1 = df['timestamp'] > p[0]
        con2 = df['timestamp'] < p[1]
        if con is None:
            con = con1 & con2
        else:
            con = con | (con1 & con2)
    return df[con]

def extract_df2(df: pd.DataFrame, normal_peroids):
    con=None
    for p in normal_peroids:
        con1 = df['start_time'] > p[0]
        con2 = df['start_time'] < p[1]
        if con is None:
            con = con1 & con2
        else:
            con = con | (con1 & con2)
    return df[con]


if __name__ == '__main__':
    label_df = pd.read_csv("gaia.csv")
    label_df['st_time'] = label_df['st_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    label_df['ed_time'] = label_df['ed_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    idx = label_df[label_df['anomaly_type'].str.contains('login failure')].index.values.tolist()[0]

    trace_df = pd.read_csv("trace.csv")
    # log_df = pd.read_csv("log.csv")

    # sample normal data
    normal_peroids=[]
    normal_window=40*60*1000
    for idx in [idx]:
        st_time = label_df.loc[idx, 'st_time']
        normal_peroids.append([st_time-normal_window, st_time])
    # normal_logs=extract_df(log_df, normal_peroids)
    normal_traces=extract_df2(trace_df, normal_peroids)

    # Parallel Processing
    
    # reduce the IO count
    metric_fs = {}
    for f in os.listdir('metric'):
        if f.startswith("zookeeper") or f.startswith("system"):
            continue
        metric_fs[f] = pd.read_csv(f'metric/{f}')

    results = []
    # with mp.Pool() as pool:
    #     for f_name, metric_f in metric_fs.items():
    #         metric_name = f_name.split("_2021")[0]
    #         metric_f.rename(columns={"value": metric_name}, inplace=True)
    #         result = pool.apply_async(extract_df, [metric_f, normal_peroids])
    #         results.append(result)
    #     [result.wait() for result in results]

    #     normal_metrics = [res.get() for res in results if not res.get().empty]
    
    data={
        # "metric": normal_metrics,
        "trace": normal_traces,
        # "log": normal_logs
    }

    io_util.save('normal.pkl', data)