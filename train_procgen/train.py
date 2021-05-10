import os
from os.path import join
import json
import tensorflow as tf

from train_procgen.aug_ppo import learn

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
    num_envs = 64  # 16?
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    timesteps_per_proc = 20_000_000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', '-id', type=int, default=0)
    parser.add_argument('--use', type=str, default="randcrop")
    parser.add_argument('--arch', type=str, default="impala")
    parser.add_argument('--no_bn', dest='use_batch_norm', action='store_false')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--nupdates', type=int, default=0)
    parser.add_argument('--total_tsteps', type=int, default=0)
    parser.add_argument('--load_id', type=int, default=int(-1))
    parser.add_argument('--netrand', type=float, default=0)
    parser.set_defaults(use_batch_norm=True)

    args = parser.parse_args()
    arch = args.arch
    dropout = args.dropout
    use_batch_norm = args.use_batch_norm
    netrand = args.netrand

    
    if args.nupdates:
        timesteps_per_proc = int(args.nupdates * num_envs * nsteps)
    if not args.total_tsteps:
        args.total_tsteps = timesteps_per_proc ## use global 20_000_000 if not specified in args!

    run_ID = 'run_'+str(args.run_id).zfill(2)
    agent_str = args.use
    LOG_DIR = join("log", agent_str, "train")
    save_model = join("log", agent_str, "saved_{}_v{}.tar".format(agent_str, args.run_id) )
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
    logger.info("\nSaved args at:\n\t{}\n".format(fpath))
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
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    logger.info(venv.observation_space)
    logger.info("training")
    with sess.as_default():
        model = learn(
                agent_str=agent_str,
                use_netrand=netrand,
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
                lr=learning_rate,
                arch=arch, 
                use_batch_norm=use_batch_norm, 
                dropout=dropout,
                cliprange=clip_range,
                save_path=save_model,
                load_path=load_path,
                vf_coef=0.5,
                max_grad_norm=0.5,
                clip_vf=use_vf_clipping,
                update_fn=None,
                init_fn=None,
                comm=comm,
            )
        model.save(save_model)

if __name__ == '__main__':
    main()
