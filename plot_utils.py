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
import numpy as np
import pandas as pd
from os.path import join

def plot_compare_runs(log_dir='log/random/train', run_ids=[0,1], keys=['eprew100', 'eplenmean100'], title="random ppo agent training log"):
    pths = [join(log_dir, "run_"+str(run_id).zfill(2), 'progress.csv') for run_id in run_ids]
    args = [join(log_dir, "args_run_{}.json".format(str(run_id).zfill(2))) for run_id in run_ids]
    fig, axs = plt.subplots(1,len(keys), figsize=(7*len(keys), 5))
    for i in range(len(keys)):
        key = keys[i]
        for p, jpath, run_id in zip(pths, args, run_ids):
            with open(jpath, 'r') as jfile:
                arg = json.load(jfile)
            df = pd.read_csv(p)
            x, y = df['misc/total_timesteps'].to_numpy(), df[key].to_numpy()
            axs[i].plot(x,y, lw=1,\
                        label="run {}, level 0-{}, {}M steps".format(\
                        str(arg["run_id"]), str(arg["num_levels"]),\
                        str(arg["total_tsteps"] // int(1e6))))
        axs[i].set_title("{} history".format(key))
        axs[i].legend()
    fig.suptitle(title)
    plt.show()

def plot_test_results(agent="randcrop", load_id=0):
    ## get training info
    jpath = join("log", agent, "train", "args_run_{}.json".format(str(load_id).zfill(2)))
    with open(jpath, 'r') as jfile:
        arg = json.load(jfile)
    
    fig, axs = plt.subplots(1,1, figsize=(8, 3), sharey=True)
    for i in range(13):
        x, y = i//7, i%7
        test_p = join("log", agent, "test", "run_{}_load{}".format(\
                            str(i).zfill(2), str(load_id)))
        
        test_j = join(test_p, "args_run_{}_load{}.json".format(\
                            str(i).zfill(2), str(load_id)))
        with open(test_j, 'r') as test_jfile:
            test_arg = json.load(test_jfile)
        p = join(test_p, "progress.csv")
        df = pd.read_csv(p)
        rew = df["eprew100"].to_numpy()[1:]
        xs = df['misc/total_timesteps'].to_numpy()[1:]
        _mean = np.mean(rew)
        #axs.plot(xs, rew)
        axs.plot(xs, [_mean for x in xs], \
                    label="test on levels {}~{}".format(\
                    str(test_arg["start_level"]), str(test_arg["start_level"]+1000)))
        axs.legend()
    title = "test {} agent trained on 0~{} levels for {}M steps".format(\
             agent, str(arg["num_levels"]), str(arg["total_tsteps"] // int(1e6)))
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

