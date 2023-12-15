import numpy as np
import copy
from mdp_utils import *


class BREX:
    def __init__(env, min_reward, max_reward, chain_len, step, conf, samp_flag = 0, reject = False, num_step = 1, gamma = 0.95):
        self.env = copy.deepcopy(env)
        self.r_min = min_reward
        self.r_max = max_reward
        self.chain_length = chain_len
        self.step_size = step
        self.alpha = conf
        self.sampling_flag = samp_flag
        self.mcmc_reject = reject
        self.num_steps = num_step
        self.gamma = gamma

        # FIXME: initializeMDP()  (bound rewards)

        self.map_env = copy.deepcopy(env)
        # FIXME: and then set weights to the same new weights as from initializeMDP()
        self.map_posterior = 0

        self.r_chain = [None for _ in range(self.chain_length)]
        self.posteriors = [0.0 for _ in range(self.chain_length)]
        self.iteration = 0
    

    def run(self, epsilon):  # FIXME: include default eps
        if self.iteration > 0:
            for i in range(self.chain_length - 1):
                self.r_chain[i] = None
        self.iteration += 1
        self.map_posterior = 0
        self.r_chain[0] = self.env
        posterior = self.calc_posterior(self.env)
        # TODO: line 134
    

    def calc_posterior(self, env):
        posterior = 0
        prior = 0
        posterior += prior
        weights = env.feature_weights
        for i in range(len(self.preferences)):  # FIXME: MAKE PREFERENCES
            worse_idx, better_idx = self.preferences[i]
            fcounts_worse = self.feature_counts[worse_idx]
            fcounts_better = self.feature_counts[better_idx]
            worse_return = np.dot(weights, fcounts_worse)
            better_return = np.dot(weights, fcounts_better)
            z = (self.alpha * better_return, self.alpha * worse_return)
            likelihood = self.alpha * better_return - logsumexp(z)
            posterior += likelihood
        return posterior