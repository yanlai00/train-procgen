import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from train_procgen.policies import RandomCnnPolicy, CnnPolicy, EnsembleCnnPolicy, impala_cnn, random_impala_cnn, cut_impala_cnn, CrossCnnPolicy, CutoutCnnPolicy
USE_COLOR_TRANSFORM = 0
from .utils import observation_input, sf01, constfn, safemean
from .models import BaseModel
from .runner import Runner
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.models import build_impala_cnn
from baselines.common.policies import build_policy
from baselines import logger
from mpi4py import MPI

from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from baselines.common.distributions import make_pdtype

FM_COEFF = 0.002
REAL_THRES = 0.1

BOT_NORM = np.array([[[0.15532861, 0.15149334, 0.14286397],
        [0.16204034, 0.15916389, 0.14478161],
        [0.18217553, 0.17738144, 0.15820507]],

       [[0.16683444, 0.16683444, 0.16683444],
        [0.1802579 , 0.18121672, 0.1754638 ],
        [0.15341098, 0.15341098, 0.14190515]],

       [[0.16012271, 0.1649168 , 0.17066971],
        [0.15820507, 0.16204034, 0.16299916],
        [0.16971089, 0.17450499, 0.16875207]],

       [[0.16587562, 0.1754638 , 0.18505199],
        [0.15532861, 0.16395798, 0.16875207],
        [0.17929908, 0.18888727, 0.18984608]]])

def recenter(obs, bot_norm):
    """
    Takes in original obs and recenter to agent-centric
    """
    recentered = []
    for i in range(obs.shape[0]): ## iterate through envs
        one_obs = obs[i]
        piece = np.transpose(one_obs[:][57:57+4][:], (1,0,2))
        bg = np.zeros((64*2, 64, 3), dtype=np.uint8)
        diff_min = 100
        loc_min = -1
        for loc in range(64-3):
            block = piece[loc:loc+3]
            block = np.transpose(block, (1,0,2))
            diff = abs(np.sum(bot_norm - block/np.linalg.norm(block)))
            if diff < diff_min:
                diff_min = diff
                loc_min = loc
        blk = piece[loc_min:loc_min+3]
        #print("detected loc: ", loc_min)
        center = 62 - loc_min
        bg[center:center+64] = one_obs.transpose((1,0,2))
        recentered.append(bg.transpose((1,0,2)))
    recentered = np.array(recentered)
    return recentered

def vanilla(obs, shape=(32, 32)): ##obs.shape==320,64,64,3
    return obs

def crosscut(obs, shape=(32, 32)): ##obs.shape==320,64,64,3
    x = np.random.randint(46)
    y = np.random.randint(64)
    obs[:, :, y:y+3,:] = 0
    obs[:, x:x+4,:,:] = 0
    return obs

def cutout(obs, shape=(32, 32)): ##obs.shape==320,64,64,3
    x = np.random.randint(46)
    obs[:, x:x+4,:,:] = 0
    return obs

def jitter(obs):
    i = np.random.randint(3)
    jitter = np.random.randint(0,50,(64,64), dtype=np.uint8)
    obs[:,:,:,i] += jitter
    return obs

def randcrop(obs, shape=(32, 32)): ##obs.shape==320,64,64,3
    """
    Takes in original obs and randomly crop 
    to a 55x55 
    """
    x, y = np.random.randint(15,size=(2,))
    obs[:, :, :y,:] = 0
    obs[:, :, 64-y:, :] = 0
    return obs

def augment(obs):
    cpy1 = crosscut(obs.copy())
    cpy2 = randcrop(obs.copy())
    cpy3 = cutout(obs.copy())
    return np.concatenate([cpy1, cpy2, cpy3, obs], axis=0)


def learn(*, network, sess, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, save_path=None, load_path=None, **network_kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    
    nbatch_train = nbatch // nminibatches
    policy = CrossCnnPolicy # TODO
    model = BaseModel(policy=policy, sess=sess, ob_space=ob_space, ac_space=ac_space, 
        nbatch_act=nenvs, nbatch_train=nbatch_train,
        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)
        logger.info("Model pramas loaded from save")
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, aug_func=vanilla) # TODO
    logger.info("Initilizing runner")
    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    tfirststart = time.time()
    active_ep_buf = epinfobuf100

    nupdates = total_timesteps//nbatch
    logger.info("Running {} updates, each needs {} batches".format(nupdates, nbatch))
    mean_rewards = []
    datapoints = []

    run_t_total = 0
    train_t_total = 0

    can_save = True
    checkpoints = list(range(0,2049,10))
    saved_key_checkpoints = [False] * len(checkpoints)

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        run_tstart = time.time()
        
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        #logger.info('rollouts complete')

        mblossvals = []

        logger.info('update: {} updating parameters...'.format(update))
        train_tstart = time.time()
        
        if states is None:
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                        
        else:
            assert nenvs % nminibatches == 0
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # update the dropout mask
        sess.run([model.train_model.dropout_assign_ops])

        train_elapsed = time.time() - train_tstart
        train_t_total += train_elapsed

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            step = update*nbatch

            rew_mean_10 = safemean([epinfo['r'] for epinfo in epinfobuf10])
            rew_mean_100 = safemean([epinfo['r'] for epinfo in epinfobuf100])
            ep_len_mean_10 = np.nanmean([epinfo['l'] for epinfo in epinfobuf10])
            ep_len_mean_100 = np.nanmean([epinfo['l'] for epinfo in epinfobuf100])
            
            logger.info('\n----', update)

            mean_rewards.append(rew_mean_10)
            datapoints.append([step, rew_mean_10])
            mean_rewards.append(rew_mean_10)
            logger.logkv('eprew10', rew_mean_10)
            logger.logkv('eprew100', rew_mean_100)
            logger.logkv('eplenmean10', ep_len_mean_10)
            logger.logkv('eplenmean100', ep_len_mean_100)
            logger.logkv('nupdate', update)

            logger.logkv('misc/total_time_elapsed', tnow - tfirststart)
            logger.logkv('misc/run_t_total', run_t_total)
            logger.logkv('misc/train_t_total', train_t_total)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("fps", fps)

            if len(mblossvals):
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.info(lossname, lossval)
                    #tb_writer.log_scalar(lossval, lossname)
                    logger.logkv('loss/' + lossname, lossval)
            logger.info('----\n')
            logger.dumpkvs()

    if save_path:
        model.save(save_path)

    env.close()
    return model