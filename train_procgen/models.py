import tensorflow as tf
from baselines import logger
import numpy as np
import joblib
from baselines.common.tf_util import initialize

L2_WEIGHT = 1e-5
FM_COEFF = 0.002

class BaseModel(object):
    def __init__(self, *, sess, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, arch, use_batch_norm, dropout):
        logger.info("nbatch_train in Model init: ", nbatch_train)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, arch, use_batch_norm, dropout)
        logger.info("ac+obs space at model init: ", ob_space.shape, ac_space.shape)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, arch, use_batch_norm, dropout)


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

        params = tf.trainable_variables()
        weight_params = [v for v in params if '/b' not in v.name]

        total_num_params = 0

        for p in params:
            shape = p.get_shape().as_list()
            num_params = np.prod(shape)
            # mpi_print('param', p, num_params)
            total_num_params += num_params

        logger.info('total num params:', total_num_params)

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * L2_WEIGHT

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        grads_and_var = trainer.compute_gradients(loss, params)

        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        _train = trainer.apply_gradients(grads_and_var)

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
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        initialize()

class RandomModel(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, arch, use_batch_norm, dropout):
        sess = tf.compat.v1.get_default_session()
        
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, arch, use_batch_norm, dropout)
            
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, arch, use_batch_norm, dropout)

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
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.clean_train = clean_train

        initialize()