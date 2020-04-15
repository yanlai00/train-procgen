import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import argparse
import csv
import os
import json
import sys
import pickle
import pandas as pd
from os.path import join
from collections import defaultdict, deque
import itertools


def plot_compare_runs(log_dir='log/random_log/train', run_ids=[0], keys=['eprew', 'eplenmean'], title="random ppo agent training log"):
    pths = [join(log_dir, "run_"+str(run_id).zfill(2), 'progress.csv') for run_id in run_ids]
    fig, axs = plt.subplots(1,len(keys), figsize=(5*len(keys), 5))
    for i in range(len(keys)):
        key = keys[i]
        for p in pths:
            df = pd.read_csv(p)
            x, y = df['misc/total_timesteps'].to_numpy(), df[key].to_numpy()
            axs[i].plot(x,y)
        axs[i].set_title("{} history".format(key))
    fig.suptitle(title)
    plt.show()


def plot_compare_agents(log_dirs=['log/random_log/test', 'log/vanilla/test'], \
                        legends=["random agent", "basic agent"],
                        run_ids=[1,1], keys=['eprew100', 'eplenmean'],
                        title="Comparing two agents performance on unseen levels"):
    pths = []
    for i in range(len(log_dirs)):
        pths.append(join(log_dirs[i], "run_"+str(run_ids[i]).zfill(2), 'progress.csv'))
    fig, axs = plt.subplots(1,len(keys), figsize=(5*len(keys), 5))
    for i in range(len(keys)):
        key = keys[i]
        for p in pths:
            df = pd.read_csv(p)
            x, y = df['misc/total_timesteps'].to_numpy(), df[key].to_numpy()
            axs[i].plot(x,y)
        axs[i].legend(legends)
        axs[i].set_title("{} history".format(key))
    fig.suptitle(title)
    plt.show()