import numpy as np

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

def recenter(obs):
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
            diff = abs(np.sum(BOT_NORM - block/np.linalg.norm(block)))
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

def gray_scale(obs):
    # imgs: b x c x h x w
    b, c, h, w = obs.shape
    frames = c // 3

    imgs = obs.view([b, frames, 3, h, w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * np.ones([1, 1, 3, 1, 1], dtype=imgs.dtype)  # broadcast tiling
    return imgs
