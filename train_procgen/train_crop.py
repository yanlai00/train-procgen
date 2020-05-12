"""
Train an 
"""
import os
from os.path import join
import json
import tensorflow as tf

# from baselines.ppo2 import ppo2
import cutout_ppo
import crop_ppo
import cross_ppo
import randcuts_ppo

PPO_FUNCs = {
    "cutout": cutout_ppo,
    "randcuts": randcuts_ppo,
    "cross": cross_ppo,
    "randcrop": crop_ppo
    }

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
    timesteps_per_proc = 30_000_000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', '-id', type=int, default=0)
    parser.add_argument('--use', type=str, default="randcrop")
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--nupdates', type=int, default=0)
    parser.add_argument('--total_tsteps', type=int, default=0)
    parser.add_argument('--load_id', type=int, default=int(-1))

    args = parser.parse_args()
    
    if args.nupdates:
        timesteps_per_proc = int(args.nupdates * num_envs * nsteps)
    if not args.total_tsteps:
        args.total_tsteps = timesteps_per_proc ## use global 20_000_000 if not specified in args!

    run_ID = 'run_'+str(args.run_id).zfill(2)
    ## select which ppo to use:
    agent_str = args.use
    LOG_DIR = join("log", agent_str, "train")
    save_model = join("log", agent_str, "saved_{}_v{}.tar".format(agent_str, args.run_id) )
    ppo_func = PPO_FUNCs[agent_str]
    load_path = None
    if args.load_id > -1:
        load_path =  join("log", agent_str, "saved_{}_v{}.tar".format(agent_str, args.load_id) )


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

    logpath = join(LOG_DIR, run_ID)
    if not os.path.exists(logpath):
        os.system("mkdir -p %s" % logpath)
    logger.configure(dir=logpath, format_strs=format_strs)

    fpath = join(LOG_DIR, 'args_{}.json'.format(run_ID))
    with open(fpath, 'w') as fh:
        json.dump(vars(args), fh, indent=4, sort_keys=True)
    print("\nSaved args at:\n\t{}\n".format(fpath))

    logger.info("\n Saving model to file {}".format(save_model))
    
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
    config = tf.compat.v1.ConfigProto(log_device_placement=True)#device_count={'GPU':0, 'XLA_GPU':0})
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.compat.v1.Session(config=config)
    #sess.__enter__()

    logger.info(venv.observation_space)
    logger.info("training")
    with sess.as_default():
        model = ppo_func.learn(
                sess=sess,
                env=venv,
                network=None,
                total_timesteps=args.total_tsteps,
                save_interval=1000,
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
                save_path=save_model,
                load_path=load_path,
                vf_coef=0.5,
                max_grad_norm=0.5,
            )
        model.save(save_model)

if __name__ == '__main__':
    main()
