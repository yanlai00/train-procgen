"""
NOTE: for debug purposes, need to reset clean_loss to match baseline.ppo2! 
    aka get rid of l2_loss and fm_losss
Taking Kimin's netrand code https://github.com/pokaxpoka/netrand/blob/master/sources/random_ppo2.py
and replacing openai/coinrun Configs with hard-coded params to fit openai/procgen env
"""

import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from train_procgen.policies import RandomCnnPolicy, CnnPolicy
from policies import random_impala_cnn
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.models import build_impala_cnn
from baselines.common.policies import build_policy
from baselines import logger
from mpi4py import MPI
from .utils import observation_input, sf01, constfn

from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from baselines.common.distributions import make_pdtype

## mentioned in paper imported from Config
DROPOUT = 0.0
L2_WEIGHT = 1e-5
FM_COEFF = 1e-3
REAL_THRES = 0.9

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.compat.v1.get_default_session()
        
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps)
            
        norm_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, nsteps=1)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # clean training
        clean_neglogpac = train_model.clean_pd.neglogp(A)
        clean_entropy = tf.reduce_mean(train_model.clean_pd.entropy())
        
        clean_vpred = train_model.clean_vf
        clean_vpredclipped = OLDVPRED + tf.clip_by_value(train_model.clean_vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        clean_vf_losses1 = tf.square(clean_vpred - R)
        clean_vf_losses2 = tf.square(clean_vpredclipped - R)
        clean_vf_loss = .5 * tf.reduce_mean(tf.maximum(clean_vf_losses1, clean_vf_losses2))
        clean_ratio = tf.exp(OLDNEGLOGPAC - clean_neglogpac)
        clean_pg_losses = -ADV * clean_ratio
        clean_pg_losses2 = -ADV * tf.clip_by_value(clean_ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        clean_pg_loss = tf.reduce_mean(tf.maximum(clean_pg_losses, clean_pg_losses2))
        clean_approxkl = .5 * tf.reduce_mean(tf.square(clean_neglogpac - OLDNEGLOGPAC))
        clean_clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(clean_ratio - 1.0), CLIPRANGE)))
        
        # FM loss
        fm_loss = tf.compat.v1.losses.mean_squared_error(
            labels=tf.stop_gradient(train_model.CH), predictions=train_model.H)
        
        params = tf.compat.v1.trainable_variables()
        weight_params = [v for v in params if '/b' not in v.name]

        total_num_params = 0

        for p in params:
            shape = p.get_shape().as_list()
            num_params = np.prod(shape)
            total_num_params += num_params

        logger.info('total num params:', total_num_params)

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * L2_WEIGHT  + fm_loss * FM_COEFF
        clean_loss = clean_pg_loss - clean_entropy * ent_coef + clean_vf_loss * vf_coef + fm_loss * FM_COEFF + l2_loss * L2_WEIGHT
        

        trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        grads_and_var = trainer.compute_gradients(loss, params)
        clean_grads_and_var = trainer.compute_gradients(clean_loss, params)

        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        
        clean_grads, clean_var = zip(*clean_grads_and_var)
        if max_grad_norm is not None:
            clean_grads, _grad_norm = tf.clip_by_global_norm(clean_grads, max_grad_norm)
        clean_grads_and_var = list(zip(clean_grads, clean_var))
        
        _train = trainer.apply_gradients(grads_and_var)
        _clean_train = trainer.apply_gradients(clean_grads_and_var)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values

            adv_mean = np.mean(advs, axis=0, keepdims=True)
            adv_std = np.std(advs, axis=0, keepdims=True)
            advs = (advs - adv_mean) / (adv_std + 1e-8)

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, l2_loss, fm_loss, _train],
                td_map
            )[:-1]
        
        def clean_train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values

            adv_mean = np.mean(advs, axis=0, keepdims=True)
            adv_std = np.std(advs, axis=0, keepdims=True)
            advs = (advs - adv_mean) / (adv_std + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [clean_pg_loss, clean_vf_loss, clean_entropy, clean_approxkl, clean_clipfrac, l2_loss, fm_loss, _clean_train],
                td_map
            )[:-1]
        
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'l2_loss', 'fm_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value_with_clean
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.clean_train = clean_train
        self.step_with_clean = act_model.step_with_clean

        # if Config.SYNC_FROM_ROOT:
        #     if MPI.COMM_WORLD.Get_rank() == 0:
        #         initialize()
            
        #     global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        #     sync_from_root(sess, global_variables) #pylint: disable=E1101
        # else:
        #     initialize()
        initialize()

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        

    def run(self, clean_flag):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs \
            = self.model.step_with_clean(clean_flag, self.obs, self.states, self.dones)
            
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
        last_values = self.model.value(clean_flag, self.obs, self.states, self.dones)

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


def learn(*, network, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, save_path=None, load_path=None, **network_kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    sess = tf.get_default_session()
    # tb_writer = TB_Writer(sess)

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
    policy = RandomCnnPolicy
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, 
        nbatch_act=nenvs, nbatch_train=nbatch_train,
        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm)

    # utils.load_all_params(sess)
    if load_path is not None:
        model.load(load_path)
        logger.info("Model pramas loaded from save")
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    
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
    init_rand = tf.variables_initializer([v for v in tf.global_variables() if 'randcnn' in v.name])

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        #logger.info('collecting rollouts...')
        run_tstart = time.time()
        sess.run(init_rand) # re-initialize the parameters of random networks
        clean_flag = np.random.rand(1)[0] > REAL_THRES ##
        ## NOTE: for sanity check (aka always run clean), do clean_flag = 1
        ## for debugged (aka always run perturbed), do 
        # clean_flag = 1
        #logger.info("\n clean_flag set to "+str(clean_flag))
        
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(clean_flag)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed

        mblossvals = []

        #logger.info('updating parameters...')
        train_tstart = time.time()
        
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    
                    if clean_flag:
                        mblossvals.append(model.clean_train(lrnow, cliprangenow, *slices))
                    else:
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                        
        else: # recurrent version
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
        #logger.info('update complete')

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            step = update*nbatch
            #rew_mean_10 = utils.process_ep_buf(active_ep_buf, tb_writer=tb_writer, suffix='', step=step)

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

            
            #logger.info('time_elapsed', tnow - tfirststart, run_t_total, train_t_total)
            logger.logkv('misc/total_time_elapsed', tnow - tfirststart)
            logger.logkv('misc/run_t_total', run_t_total)
            logger.logkv('misc/train_t_total', train_t_total)

            #logger.info('timesteps', update*nsteps, total_timesteps)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            
            #logger.info('fps', fps)
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

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)