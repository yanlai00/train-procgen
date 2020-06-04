"""
In Model: two "parallel" impala cnns that can be train separately via train1 and train2
In EnsembleCnnPolicy: takes in flag to decide which cnn to use to step/get value
"""

import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from train_procgen.policies import RandomCnnPolicy, CnnPolicy
USE_COLOR_TRANSFORM = 0
## NOTE: subclass this instead of standard CNN!!
from policies import random_impala_cnn
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.models import build_impala_cnn
from baselines.common.policies import build_policy
from baselines import logger
from mpi4py import MPI

# from coinrun.tb_utils import TB_Writer
# import coinrun.main_utils as utils
# from coinrun.config import Config

#logger.info = utils.logger.info

from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from baselines.common.distributions import make_pdtype

## mentioned in paper imported from Config
DROPOUT = 0.0
L2_WEIGHT = 1e-5
FM_COEFF = 0.1
ALT_THRES = 0.5 # clean inputs are used with this probability α  
## Deleted MPIAdamoptimizer, needed when comm.Get_size() > 1 (rn ==1)
def impala_cnn(images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    #use_batch_norm = Config.USE_BATCH_NORM == 1 NOTE: Should prob. use this???
    use_batch_norm = True

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out): ## NO dropout for now 
        if DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)
        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)
        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, dropout_assign_ops

def observation_input(ob_space, batch_size=None, name='Ob'):
    from gym.spaces import Discrete, Box, MultiDiscrete
    from baselines.common.input import encode_observation
    """
    Modified from baselines to reshape input obs
    """
    assert isinstance(ob_space, Discrete) or isinstance(ob_space, Box) or isinstance(ob_space, MultiDiscrete), \
        'Baselines only deal with Discrete and Box observation spaces'
    dtype = ob_space.dtype
    if dtype == np.int8:
        dtype = np.uint8
    shape = (ob_space.shape[0], ob_space.shape[1], ob_space.shape[2])
    #shape = (64, 64*2, 3)
    placeholder = tf.placeholder(shape=(batch_size,) + shape, dtype=dtype, name=name)
    return placeholder, encode_observation(ob_space, placeholder)

class EnsembleCnnPolicy(CnnPolicy): ## Not considering color_transform!
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        with tf.variable_scope("model1", reuse=tf.AUTO_REUSE):
            processed_x3 = processed_x
            h1, self.dropout_assign_ops1 = impala_cnn(processed_x3)
            vf1 = fc(h1, 'v1', 1)[:,0]
            self.pd1, self.pi1 = self.pdtype.pdfromlatent(h1, init_scale=0.01)

        with tf.variable_scope("model2", reuse=tf.AUTO_REUSE):
            processed_x3 = processed_x
            h2, self.dropout_assign_ops2 = impala_cnn(processed_x3)
            vf2 = fc(h2, 'v2', 1)[:,0]
            self.pd2, self.pi2 = self.pdtype.pdfromlatent(h2, init_scale=0.01)

        a0_1 = self.pd1.sample()
        neglogp0_1 = self.pd1.neglogp(a0_1)

        a0_2 = self.pd2.sample()
        neglogp0_2 = self.pd1.neglogp(a0_2)


        self.initial_state = None
        
        def alt_step(flag, ob, *_args, **_kwargs):
            a1, v1, neglogp1, a2, v2, neglogp2 \
            = sess.run([a0_1, vf1, neglogp0_1, a0_2, vf2, neglogp0_2], {X:ob})
            if flag:
                return a2, v2, self.initial_state, neglogp2
            else:
                return a1, v1, self.initial_state, neglogp1
        
        def alt_value(flag, ob, *_args, **_kwargs):
            v1, v2 = sess.run([vf1, vf2], {X:ob})
            if flag:
                return v2
            else:
                return v1

        self.X = X

        self.H1 = h1
        self.vf1 = vf1
        
        self.H2 = h2
        self.vf2 = vf2

        self.alt_step = alt_step
        self.alt_value = alt_value

    
class Model(object):
    def __init__(self, *, sess, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        # sess = tf.get_default_session()
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps)
        norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac1 = train_model.pd1.neglogp(A)
        entropy1 = tf.reduce_mean(train_model.pd1.entropy())

        vpred1 = train_model.vf1
        vpredclipped1 = OLDVPRED + tf.clip_by_value(train_model.vf1 - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1_1 = tf.square(vpred1 - R)
        vf_losses2_1 = tf.square(vpredclipped1 - R)
        vf_loss1 = .5 * tf.reduce_mean(tf.maximum(vf_losses1_1, vf_losses2_1))
        ratio1 = tf.exp(OLDNEGLOGPAC - neglogpac1)
        pg_losses_1 = -ADV * ratio1
        pg_losses2_1 = -ADV * tf.clip_by_value(ratio1, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss1 = tf.reduce_mean(tf.maximum(pg_losses_1, pg_losses2_1))
        approxkl1 = .5 * tf.reduce_mean(tf.square(neglogpac1 - OLDNEGLOGPAC))
        clipfrac1 = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio1 - 1.0), CLIPRANGE)))

        ## A whole different set
        neglogpac2 = train_model.pd2.neglogp(A)
        entropy2 = tf.reduce_mean(train_model.pd2.entropy())

        vpred2 = train_model.vf2
        vpredclipped2 = OLDVPRED + tf.clip_by_value(train_model.vf2 - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1_2 = tf.square(vpred2 - R)
        vf_losses2_2 = tf.square(vpredclipped2 - R)
        vf_loss2 = .5 * tf.reduce_mean(tf.maximum(vf_losses1_2, vf_losses2_2))
        ratio2 = tf.exp(OLDNEGLOGPAC - neglogpac2)
        pg_losses_2 = -ADV * ratio2
        pg_losses2_2 = -ADV * tf.clip_by_value(ratio2, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss2 = tf.reduce_mean(tf.maximum(pg_losses_2, pg_losses2_2))
        approxkl2 = .5 * tf.reduce_mean(tf.square(neglogpac2 - OLDNEGLOGPAC))
        clipfrac2 = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio2 - 1.0), CLIPRANGE)))


        fm_loss = tf.compat.v1.losses.mean_squared_error(
            labels=tf.stop_gradient(train_model.H1), predictions=train_model.H2)

        params = tf.trainable_variables()
        weight_params1 = [v for v in params if '/b' not in v.name]
        weight_params2 = weight_params1

        total_num_params = 0
        for p in params:
            shape = p.get_shape().as_list()
            num_params = np.prod(shape)
            # mpi_print('param', p, num_params)
            total_num_params += num_params

        logger.info('total num params:', total_num_params)

        l2_loss1 = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params1])
        l2_loss2 = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params2])

        loss1 = pg_loss1 - entropy1 * ent_coef + vf_loss1 * vf_coef + l2_loss1 * L2_WEIGHT + fm_loss * FM_COEFF
        loss2 = pg_loss2 - entropy2 * ent_coef + vf_loss2 * vf_coef + l2_loss2 * L2_WEIGHT + fm_loss * FM_COEFF

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        grads_and_var1 = trainer.compute_gradients(loss1, params)
        grads_and_var2 = trainer.compute_gradients(loss2, params)

        grads1, var1 = zip(*grads_and_var1)
        if max_grad_norm is not None:
            grads1, _grad_norm1 = tf.clip_by_global_norm(grads1, max_grad_norm)
        grads_and_var1 = list(zip(grads1, var1))

        grads2, var2 = zip(*grads_and_var2)
        if max_grad_norm is not None:
            grads2, _grad_norm2 = tf.clip_by_global_norm(grads2, max_grad_norm)
        grads_and_var2 = list(zip(grads2, var2))

        _train1 = trainer.apply_gradients(grads_and_var1)
        _train2 = trainer.apply_gradients(grads_and_var2)

        def train1(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
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
                [pg_loss1, vf_loss1, entropy1, approxkl1, clipfrac1, l2_loss1, fm_loss, _train1],
                td_map
            )[:-1]

        def train2(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
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
                [pg_loss2, vf_loss2, entropy2, approxkl2, clipfrac2, l2_loss2, fm_loss, _train2],
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

        self.train1 = train1
        self.train2 = train2
        self.train_model = train_model
        self.act_model = act_model

        self.alt_step = act_model.alt_step
        self.alt_value = act_model.alt_value

        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        initialize()

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        
    def run(self, alt_flag):
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

            # Take actions in env and look at the results
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

 

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f
    
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

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
    policy = EnsembleCnnPolicy
    model = Model(policy=policy, sess=sess, ob_space=ob_space, ac_space=ac_space, 
        nbatch_act=nenvs, nbatch_train=nbatch_train,
        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm)

    # utils.load_all_params(sess)
    if load_path is not None:
        model.load(load_path)
        logger.info("Model pramas loaded from save")
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    logger.info("Initilizing runner")
    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    tfirststart = time.time()
    active_ep_buf = epinfobuf100

    nupdates = total_timesteps // nbatch
    logger.info("Running {} updates, each needs {} batches".format(nupdates, nbatch))
    mean_rewards = []
    datapoints = []

    run_t_total = 0
    train_t_total = 0

    can_save = True
    checkpoints = list(range(0,2049,10))
    saved_key_checkpoints = [False] * len(checkpoints)
    #init_rand = tf.variables_initializer([v for v in tf.global_variables() if 'randcnn' in v.name])

    # if Config.SYNC_FROM_ROOT and rank != 0:
    #     can_save = False

    # def save_model(base_name=None):
    #     base_dict = {'datapoints': datapoints}
    #     utils.save_params_in_scopes(
    #         sess, ['model'], Config.get_save_file(base_name=base_name), base_dict)

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        logger.info('collecting rollouts...')
        run_tstart = time.time()
        alt_flag = np.random.rand(1)[0] > ALT_THRES 
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(alt_flag)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        logger.info('rollouts complete')

        mblossvals = []

        logger.info('update: {} updating parameters... for model {}'.format(update, alt_flag+1))
        train_tstart = time.time()
        
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    if alt_flag:
                        mblossvals.append(model.train2(lrnow, cliprangenow, *slices))
                    else:
                        mblossvals.append(model.train1(lrnow, cliprangenow, *slices))
                        
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
                    if alt_flag:
                        mblossvals.append(model.train2(lrnow, cliprangenow, *slices, mbstates))
                    else:
                        mblossvals.append(model.train1(lrnow, cliprangenow, *slices, mbstates))

        # update the dropout mask
        sess.run([model.train_model.dropout_assign_ops1])
        sess.run([model.train_model.dropout_assign_ops2])

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
            logger.logkv('nupdate', update)

            
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
 
        # if save_interval and (update % save_interval == 0):
        #     _path = save_path[:-5]+"_"+str(update)+".tar"
        #     model.save(_path)
        #if can_save:
        if 0: ## not doing checkpoint saving yet
            if save_interval and (update % save_interval == 0):
                save_model()

            for j, checkpoint in enumerate(checkpoints):
                if (not saved_key_checkpoints[j]) and (step >= (checkpoint * 1e6)):
                    saved_key_checkpoints[j] = True
                    save_model(str(checkpoint) + 'M')

    # save_model()
    if save_path:
        model.save(save_path)

    env.close()
    return model
