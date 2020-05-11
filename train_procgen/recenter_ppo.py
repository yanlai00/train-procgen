"""
Change based on random_ppo but with recenter
"""

import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from collections import deque
from train_procgen.policies import RandomCnnPolicy ## NOTE: subclass this instead of standard CNN!!
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

## (4, 3, 3) normalized bot patch for recentering 
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

## mentioned in paper imported from Config
DROPOUT = 0.0
L2_WEIGHT = 1e-5
FM_COEFF = 0.002
REAL_THRES = 0.1 # clean inputs are used with this probability α  
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

    def dropout_layer(out):
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
    # Original: placeholder = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)
    #shape = (ob_space.shape[0], ob_space.shape[1], ob_space.shape[2])
    shape = (64, 64*2, 3)
    placeholder = tf.placeholder(shape=(batch_size,) + shape, dtype=dtype, name=name)
    return placeholder, encode_observation(ob_space, placeholder)

class RecenterCnnPolicy(RandomCnnPolicy): ## Not considering color_transform!
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        ## X:Tensor("Ob:0", shape=(320, 64, 64, 3), dtype=uint8)
        # Tensor("ToFloat:0", shape=(320, 64, 64, 3), dtype=float32)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        mc_index = tf.placeholder(tf.int64, shape=[1], name='mc_index')
        
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):    
            h, self.dropout_assign_ops = random_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            clean_h, _ = impala_cnn(scaled_images)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h, init_scale=0.01)
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
            
        def value_with_clean(flag, ob, *_args, **_kwargs):
            v, c_v = sess.run([vf, clean_vf], {X:ob})
            if flag:
                return c_v
            else:
                return v
            
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf =clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean
        self.value_with_clean = value_with_clean
        
        
class Model(object):
    def __init__(self, *, sess, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        # sess = tf.get_default_session()
        
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
        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE), tf.float32) )

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
            ## Comment this out to see all params
            #logger.info('param', p, num_params)
            total_num_params += num_params

        logger.info('total num params:', total_num_params)

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * L2_WEIGHT + fm_loss * FM_COEFF
        clean_loss = clean_pg_loss - clean_entropy * ent_coef + clean_vf_loss * vf_coef + l2_loss * L2_WEIGHT #+ fm_loss * FM_COEFF

        #if Config.SYNC_FROM_ROOT:
        if 0:
            trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
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
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, l2_loss, _train],
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
                [clean_pg_loss, clean_vf_loss, clean_entropy, clean_approxkl, clean_clipfrac, l2_loss, _clean_train],
                td_map
            )[:-1]
        
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'l2_loss']

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
    # recentered = []
    # obs = tf.constant(obs, dtype=tf.float32)
    # for i in range(obs.shape[0]): ## iterate through envs
    #     one_obs = obs[i]
    #     piece = tf.transpose(one_obs[:][57:57+4][:], perm=[1,0,2])
    #     bg = tf.Variable(tf.zeros((64*2, 64, 3), dtype=tf.float32))
    #     diff_min = 100
    #     loc_min = -1
    #     for loc in range(64-3):
    #         block = piece[loc:loc+3]
    #         block = tf.transpose(block, perm=[1,0,2])
    #         diff = tf.math.abs(tf.reduce_sum(bot_norm - block/tf.norm(block)))
    #         evl = diff.eval() 
    #         if evl < diff_min:
    #             diff_min = evl
    #             loc_min = loc
    #     blk = piece[loc_min:loc_min+3]
    #     #print("detected loc: ", loc_min)
    #     center = 62 - loc_min
    #     bg[center:center+64].assign(tf.transpose(one_obs, perm=[1,0,2]))
    #     recentered.append(tf.transpose(bg, perm=[1,0,2]))
    # recentered = tf.constant(recentered)
    # return recentered


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.obs = recenter(self.obs, BOT_NORM)
        # plt.imsave("recentered0.jpg", self.obs[0]) Sanity check!

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
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            self.obs = recenter(self.obs, BOT_NORM)
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


def learn(*, network, sess, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, save_path=None, load_path=None, **network_kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    #sess = tf.get_default_session()
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
    policy = RecenterCnnPolicy
    model = Model(policy=policy, sess=sess, ob_space=ob_space, ac_space=ac_space, 
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

        #logger.info('collecting rollouts...')
        run_tstart = time.time()
        sess.run(init_rand) # re-initialize the parameters of random networks
        # clean_flag = np.random.rand(1)[0] > REAL_THRES NOTE: always use 1 for now!
        clean_flag = 1
        
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(clean_flag)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        #logger.info('rollouts complete')

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
    return mean_rewards

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)