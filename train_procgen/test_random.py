import os
from os.path import join
import json
import numpy as np 
import tensorflow as tf
# from baselines.ppo2 import ppo2

from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

## random_ppo imports

import train_procgen
from train_procgen.policies import RandomCnnPolicy 
from train_procgen.random_ppo import Model, Runner, safemean
from collections import deque


LOG_DIR = 'log/random_log/test'
LOAD_PATH = 'log/saved_random.tar'
def main():
    num_envs = 64
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    total_timesteps = 100_000 ## now this counts steps in testing runs
    use_vf_clipping = True

    ## From random_ppo.py
    max_grad_norm = 0.5
    vf_coef=0.5
    L2_WEIGHT = 10e-4
    FM_COEFF = 0.002
    REAL_THRES = 0.1    

    parser = argparse.ArgumentParser(description='Process procgen testing arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    ## default starting_level set to 50 to test on unseen levels!
    parser.add_argument('--start_level', type=int, default=50) 
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--nrollouts', '-nroll', type=int, default=0)

    args = parser.parse_args()
    args.total_timesteps = total_timesteps
    if args.nrollouts:
        total_timesteps = int(args.nrollouts * num_envs * nsteps)
    run_ID = 'run_'+str(args.run_id).zfill(2)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_rank_weight = 0 
    num_levels = args.num_levels

    log_comm = comm.Split(0, 0)
    format_strs = ['csv', 'stdout', 'log'] if log_comm.Get_rank() == 0 else []

    logpath = join(LOG_DIR, run_ID)
    if not os.path.exists(logpath):
        os.system("mkdir -p %s" % logpath)

    fpath = join(LOG_DIR, 'args_{}.json'.format(run_ID))
    with open(fpath, 'w') as fh:
        json.dump(vars(args), fh, indent=4, sort_keys=True)
    print("\nSaved args at:\n\t{}\n".format(fpath))

    logger.configure(dir=logpath, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, 
        num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.compat.v1.Session(config=config)
    sess.__enter__()

    logger.info("Testing")
    ## Modified based on random_ppo.learn
    env = venv
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    nrollouts = total_timesteps // nbatch

    policy = RandomCnnPolicy
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, 
        nbatch_act=nenvs, nbatch_train=nbatch_train,
        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm)

    model.load(LOAD_PATH)
    logger.info("Model pramas loaded from save")
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    # tfirststart = time.time() ## Not doing timing yet
    # active_ep_buf = epinfobuf100

    mean_rewards = []
    datapoints = []
    for rollout in range(1, nrollouts+1):
        logger.info('collecting rollouts {}...'.format(rollout))
        clean_flag = 1 ## since we are testiing, disable randomization
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(clean_flag)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        rew_mean_10 = safemean([epinfo['r'] for epinfo in epinfobuf10])
        rew_mean_100 = safemean([epinfo['r'] for epinfo in epinfobuf100])
        ep_len_mean_10 = np.nanmean([epinfo['l'] for epinfo in epinfobuf10])
        ep_len_mean_100 = np.nanmean([epinfo['l'] for epinfo in epinfobuf100])

        logger.info('\n----', rollout)
        mean_rewards.append(rew_mean_10)
        logger.logkv('eprew10', rew_mean_10)
        logger.logkv('eprew100', rew_mean_100)
        logger.logkv('eplenmean10', ep_len_mean_10)
        logger.logkv('eplenmean100', ep_len_mean_100)
        logger.logkv("misc/total_timesteps", rollout*nbatch)

        logger.info('----\n')
        logger.dumpkvs()
    env.close()

    print("Rewards history: ", mean_rewards)
    return mean_rewards

if __name__ == '__main__':
    main()