"""
Train OR test an ensemble agent
$ taskset -c 0-7 python train_procgen/train_recenter.py -id 7 --num_levels 25
"""
import os
from os.path import join
import json
import tensorflow as tf

# from baselines.ppo2 import ppo2
import ensemble_ppo2
from train_procgen.ensemble_ppo2 import Model, Runner, EnsembleCnnPolicy2
from train_procgen.random_ppo import safemean
from train_procgen.crop_ppo import sf01, constfn

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

import numpy as np
from mpi4py import MPI
import argparse
from collections import deque

SAVE_PATH = "log2/ensemble2"

num_envs = 32 # NOTE: half of ensemble_ppo since we have 2 envs 
learning_rate = 5e-4
ent_coef = .01
gamma = .999
lam = .95
nsteps = 256
nminibatches = 8
ppo_epochs = 3
clip_range = .2
#timesteps_per_proc = 20_000_000
TIMESTEPS_PER_PROC = 20_000_000
use_vf_clipping = True

## for testing
TEST_START_LEVELS = [100, 1000, 5000] + [int(i * 1e4) for i in range(1, 10)]
TRAIN_END_LEVELS = [25, 50, 100]

def train(run_ID, save_path, load_path, env1, env2, sess, logger, args):
    logger.info("training")
    with sess.as_default():
        model = ensemble_ppo2.learn(
            sess=sess,
            env1=env1,
            env2=env2,
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

class TestRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        ##self.obs = rand_crop(self.obs) NO CROPPING OR CUTOUT AT TEST TIME

    def run(self, alt_flag): ## for test only, pick ONE network 
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.alt_step(alt_flag, self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.alt_value(alt_flag, self.obs, self.states, self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

"""
Run the agent on one setup env
"""
def test_one_env(alt_flag, model, start_level, num_levels, logger, args, env=None):
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
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(alt_flag)
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
    logger.info("Average reward on levels {} ~ {}: {} ".format(start_level, start_level+num_levels, mean_rewards))
    return np.mean(mean_rewards)

"""
Use this one function to launch a series of tests (varying num_levels intervals 
SAME logger across all envs
"""
def test_all(alt_flag, load_path, logger, args):
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

    model = Model(sess=sess, policy=EnsembleCnnPolicy, ob_space=ob_space, ac_space=ac_space, 
        nbatch_act=nenvs, nbatch_train=nbatch_train,
        nsteps=nsteps, ent_coef=ent_coef, vf_coef=0.5,
        max_grad_norm=0.5)
    model.load(load_path)
    logger.info("Model pramas loaded from saved model: ", load_path)

    mean_rewards = []
    ## first, test train performance
    
    mean_rewards.append(test_one_env(alt_flag, model, 0, train_end, logger, args, env=env))

    ## then, test on sampled intervals 
    for l in TEST_START_LEVELS:
        mean_rewards.append(test_one_env(alt_flag, model, l, 100, logger, args, env=None))

    logger.info("All tests finished, mean reward history: ", mean_rewards)
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
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--load_id', type=int, default=int(-1))
    parser.add_argument('--nrollouts', '-nroll', type=int, default=0)
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--use_model', type=int, default=1, help="either model #1 or #2")
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
        run_ID += '_test{}_model{}'.format(args.load_id, args.use_model)
    
    load_path = None 
    if args.load_id > -1:
        load_path = join(SAVE_PATH, args.env_name, 'saved_ensemble_v{}.tar'.format(args.load_id))

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
        logpath = join('log2/ensemble2', args.env_name, 'test', run_ID)
    else:
        logpath = join('log2/ensemble2', args.env_name, 'train', run_ID)
        save_path = join( SAVE_PATH, args.env_name, 'saved_ensemble2_v{}.tar'.format(args.run_id) )
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
        config = tf.compat.v1.ConfigProto(\
            allow_soft_placement=True,
            log_device_placement=True)# device_count={'GPU':0})
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        sess = tf.compat.v1.Session(config=config)
        logger.info("creating 2 environments")
        n_levels = int(args.num_levels / 2)
        env1 = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, 
            num_levels=n_levels, start_level=0, distribution_mode=args.distribution_mode)
        env1 = VecExtractDictObs(env1, "rgb")
        env1 = VecMonitor(
            venv=env1, filename=None, keep_buf=100,
        )
        env1 = VecNormalize(venv=env1, ob=False)

        env2 = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, 
            num_levels=n_levels, start_level=n_levels, distribution_mode=args.distribution_mode)
        env2 = VecExtractDictObs(env2, "rgb")
        env2 = VecMonitor(
            venv=env2, filename=None, keep_buf=100,
        )
        env2 = VecNormalize(venv=env2, ob=False)

        train(run_ID, save_path, load_path, env1, env2, sess, logger, args)
    else:
        use_model = args.use_model ## 1 or 2
        alt_flag = use_model - 1
        test_all(alt_flag, load_path, logger, args)

if __name__ == '__main__':
    main()
