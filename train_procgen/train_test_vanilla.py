"""
Train OR test an vanilla agent
$ taskset -c 0-7 python train_procgen/train_recenter.py -id 7 --num_levels 25
"""
import os
from os.path import join
import json
import tensorflow as tf

# from baselines.ppo2 import ppo2
import vanilla_ppo
from vanilla_ppo import Runner as TestRunner
from train_procgen.policies import CnnPolicy
from train_procgen.random_ppo import safemean
from train_procgen.crop_ppo import Model, sf01, constfn

from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.runners import AbstractEnvRunner
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
import numpy as np
from collections import deque

SAVE_PATH = "log2/vanilla"

num_envs = 64 
learning_rate = 5e-4
ent_coef = .01
gamma = .999
lam = .95
nsteps = 256
nminibatches = 8
ppo_epochs = 3
clip_range = .2
#timesteps_per_proc = 20_000_000
TIMESTEPS_PER_PROC = 10_000_000 ## default do **half** the tsteps ensemble did 
use_vf_clipping = True
DROPOUT = 0.0
L2_WEIGHT = 1e-5

## for testing
TEST_START_LEVELS = [100, 1000, 5000] + [int(i * 1e4) for i in range(1, 10)]
TRAIN_END_LEVELS = [25, 50, 100]


def train(run_ID, save_path, load_path, venv, sess, logger, args):
    logger.info("obs space:", venv.observation_space)
    logger.info("training")
    with sess.as_default():
        model = vanilla_ppo.learn(
            sess=sess,
            env=venv,
            network=None,
            total_timesteps=args.total_tsteps,
            save_interval=2, ## doesn't matter, always save in the end for now
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=args.log_interval,
            ent_coef=ent_coef,
            # clip_vf=use_vf_clipping,
            lr=learning_rate,
            cliprange=clip_range,
            # update_fn=None,
            # init_fn=None,
	        save_path=save_path,
            load_path=load_path,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        logger.info("saving model to: ", save_path)
        model.save(save_path)

"""
Run the agent on one setup env
NO alt_flag, different from ensemble_ppo
"""
def test_one_env(model, start_level, num_levels, logger, args, env=None):
    ## Modified based on random_ppo.learn
    if not env:
        venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, 
            num_levels=num_levels, start_level=start_level, distribution_mode=args.distribution_mode)
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(
            venv=venv, filename=None, keep_buf=100,
        )
        venv = VecNormalize(venv=venv, ob=False)
        env = venv
    
    runner = TestRunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    mean_rewards = []
    datapoints = []
    for rollout in range(1, args.nrollouts+1):
        logger.info('collecting rollouts {}...'.format(rollout))
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        rew_mean_10 = safemean([epinfo['r'] for epinfo in epinfobuf10])
        rew_mean_100 = safemean([epinfo['r'] for epinfo in epinfobuf100])
        ep_len_mean_10 = np.nanmean([epinfo['l'] for epinfo in epinfobuf10])
        ep_len_mean_100 = np.nanmean([epinfo['l'] for epinfo in epinfobuf100])

        logger.info('\n----', rollout)
        mean_rewards.append(rew_mean_10)
        logger.logkv('start_level', start_level)
        logger.logkv('eprew10', rew_mean_10)
        logger.logkv('eprew100', rew_mean_100)
        logger.logkv('eplenmean10', ep_len_mean_10)
        logger.logkv('eplenmean100', ep_len_mean_100)
        logger.logkv("misc/total_timesteps", rollout*args.nbatch)

        logger.info('----\n')
        logger.dumpkvs()
    env.close()
    logger.info("Average reward on levels {} ~ {}: {} ".format(start_level, start_level+100, mean_rewards))
    return np.mean(mean_rewards)

"""
Use this one function to launch a series of tests (varying num_levels intervals 
SAME logger across all envs
"""
def test_all(load_path, logger, args):
    train_end = int(args.train_level)
    config = tf.compat.v1.ConfigProto(log_device_placement=True)#device_count={'GPU':0})
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.compat.v1.Session(config=config)

    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, 
        num_levels=train_end, start_level=0, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    venv = VecNormalize(venv=venv, ob=False)

    env = venv
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    nrollouts = args.total_tsteps // nbatch
    args.nrollouts = nrollouts
    args.nbatch = nbatch

    model = Model(sess=sess, policy=CnnPolicy, ob_space=ob_space, ac_space=ac_space, 
        nbatch_act=nenvs, nbatch_train=nbatch_train,
        nsteps=nsteps, ent_coef=ent_coef, vf_coef=0.5,
        max_grad_norm=0.5)
    model.load(load_path)
    logger.info("Model pramas loaded from saved model: ", load_path)

    mean_rewards = []
    ## first, test train performance
    
    mean_rewards.append(test_one_env(model, 0, train_end, logger, args, env=env))

    ## then, test on sampled intervals 
    for l in TEST_START_LEVELS:
        mean_rewards.append(test_one_env(model, l, 100, logger, args, env=None))

    print("All tests finished, mean reward history: ", mean_rewards)
    return 



def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', '-id', type=int, default=99)
    parser.add_argument('--nupdates', type=int, default=0)
    parser.add_argument('--total_tsteps', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--load_id', type=int, default=int(-1))
    parser.add_argument('--nrollouts', '-nroll', type=int, default=0)
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--use_model', type=int, default=1, help="when testing, either use model #1 or #2")
    parser.add_argument('--train_level', type=int, default=50)

    args = parser.parse_args()
    #timesteps_per_proc
    if args.nupdates:
        timesteps_per_proc = int(args.nupdates * num_envs * nsteps)
    if not args.total_tsteps:
        args.total_tsteps = TIMESTEPS_PER_PROC ## use global 20_000_000 if not specified in args!
    if args.nrollouts:
        total_timesteps = int(args.nrollouts * num_envs * nsteps)
        
    run_ID = 'run_'+str(args.run_id).zfill(2)
    if args.test:
        args.log_interval = 1
        args.total_tsteps = 1_000_000
        run_ID += '_test{}'.format(args.load_id)
    
    load_path = None
    if args.load_id > -1:
        load_path = join(SAVE_PATH, args.env_name, 'saved_vanilla_v{}.tar'.format(args.load_id))
    
    test_worker_interval = args.test_worker_interval
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    is_test_worker = False
    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout', 'log'] if log_comm.Get_rank() == 0 else []

    if args.test:
        logpath = join('log2/vanilla', args.env_name, 'test', run_ID)
    else:
        logpath = join('log2/vanilla', args.env_name, 'train', run_ID)
        save_path = join( SAVE_PATH, args.env_name, "saved_vanilla_v{}.tar".format(args.run_id) )
        logger.info("\n Model will be saved to file {}".format(save_path))

    if not os.path.exists(logpath):
        os.system("mkdir -p %s" % logpath)
    logger.configure(dir=logpath, format_strs=format_strs)

    fpath = join(logpath, 'args_{}.json'.format(run_ID))
    with open(fpath, 'w') as fh:
        json.dump(vars(args), fh, indent=4, sort_keys=True)
    print("\nSaved args at:\n\t{}\n".format(fpath))

    
    
    logger.info("creating tf session")
    setup_mpi_gpus()

    if not args.test:
        config = tf.compat.v1.ConfigProto(log_device_placement=True)#device_count={'GPU':0})
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        sess = tf.compat.v1.Session(config=config)
        logger.info("creating environment")
        venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, 
            num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(
            venv=venv, filename=None, keep_buf=100,
        )
        venv = VecNormalize(venv=venv, ob=False)
        train(run_ID, save_path, load_path, venv, sess, logger, args)
    else:
        test_all(load_path, logger, args)

if __name__ == '__main__':
    main()
