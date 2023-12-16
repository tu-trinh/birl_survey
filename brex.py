import numpy as np
import copy
import random
from mdp_utils import *


class BREX:
    def __init__(self, env, min_reward, max_reward, chain_len, step, conf, reject = True, epsilon = 0.0001, gamma = 0.95):
        self.init_env = copy.deepcopy(env)
        self.env = copy.deepcopy(env)
        self.gt_reward = env.feature_weights
        self.r_min = min_reward
        self.r_max = max_reward
        self.chain_length = chain_len
        self.step_size = step
        self.alpha = conf
        self.mcmc_reject = reject
        self.epsilon = epsilon
        self.gamma = gamma

        bounded_weights = copy.deepcopy(self.env.feature_weights)
        for i in range(self.env.num_features):
            bounded_weights[i] = -1 / self.env.num_features
        self.env.set_rewards(bounded_weights)

        self.map_env = copy.deepcopy(self.env)
        self.map_posterior = 0

        self.feature_counts = []
        self.preferences = []
        self.r_chain = [None for _ in range(self.chain_length)]
        self.posteriors = [0.0 for _ in range(self.chain_length)]
        self.iteration = 0
        self.closenesses = []
    
    
    def generate_proposal(self, env):  # currently set to sampling flag 4 in the original B-REX
        weights = copy.deepcopy(env.feature_weights)
        dim = env.num_features

        permutation = [i for i in range(dim)]
        unsorted = dim - 1
        while unsorted > 0:
            rand_idx = np.random.randint(0, unsorted)
            temp = permutation[rand_idx]
            permutation[rand_idx] = permutation[unsorted]
            permutation[unsorted] = temp
            unsorted -= 1
        
        pairs = []
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                pairs.append((permutation[i], permutation[j]))
        
        for pair in pairs:
            direction = "c" if np.random.random() < 0.5 else "cc"
            dim1, dim2 = pair
            if abs(weights[dim1] - 0) > self.epsilon or abs(weights[dim2] - 0) > self.epsilon:
                new_vals = self.manifold_l1_step(weights[dim1], weights[dim2], direction)
                weights[dim1], weights[dim2] = new_vals[0], new_vals[1]
        return weights
    

    def manifold_l1_step(self, w1, w2, direction):
        slack = abs(w1) + abs(w2)
        c_pos = ["++", "+-", "--", "-+"]
        c_dir = [1, -1, 1, -1]
        cc_pos = ["++", "-+", "--", "+-"]
        cc_dir = [-1, 1, -1, 1]

        sign1 = w1 >= 0
        sign2 = w2 >= 0
        if sign1 and sign2:
            cycle_pos = "++"
        elif sign1 and not sign2:
            cycle_pos = "+-"
        elif not sign1 and sign2:
            cycle_pos = "-+"
        else:
            cycle_pos = "--"
        
        if direction == "c":
            for i in range(4):
                if c_pos[i] == cycle_pos:
                    cycle_idx = i
                    break
        else:
            for i in range(4):
                if cc_pos[i] == cycle_pos:
                    cycle_idx = i
                    break
        
        step_remaining = self.step_size
        while abs(step_remaining - 0) > self.epsilon:
            if direction == "c":
                cycle_dir = c_dir[cycle_idx]
            else:
                cycle_dir = cc_dir[cycle_idx]
            max_step = step_remaining
            if (cycle_dir == 1) and ((abs(w1) + cycle_dir * step_remaining) > slack):
                max_step = slack - abs(w1)
                cycle_idx = (cycle_idx + 1) % 4
            elif (cycle_dir == -1) and ((abs(w1) + cycle_dir * step_remaining) < 0):
                max_step = abs(w1)
                cycle_idx = (cycle_idx + 1) % 4
            w1 = abs(w1) + cycle_dir * max_step
            w2 = abs(w2) - cycle_dir * max_step
            step_remaining -= max_step
        
        if direction == "c":
            cycle_pos = c_pos[cycle_idx]
        else:
            cycle_pos = cc_pos[cycle_idx]
        if cycle_pos == "-+":
            w1 *= -1
        elif cycle_pos == "+-":
            w2 *= -1
        elif cycle_pos == "--":
            w1 *= -1
            w2 *= -1
        return w1, w2
    
    
    def run(self):
        if self.iteration > 0:
            for i in range(self.chain_length - 1):
                self.r_chain[i] = None
        self.iteration += 1
        self.map_posterior = 0
        self.r_chain[0] = self.env
        posterior = self.calc_posterior(self.env)
        self.posteriors[0] = np.exp(posterior)

        i = 0
        curr_dist = np.inf
        while i < self.chain_length and curr_dist > self.epsilon:
            temp_env = copy.deepcopy(self.env)
            hyp_reward = self.generate_proposal(temp_env)
            temp_env.set_rewards(hyp_reward)
            new_posterior = self.calc_posterior(temp_env)
            prob = min(1.0, np.exp(new_posterior - posterior))
            if np.random.random() < prob:
                self.env = temp_env
                posterior = new_posterior
                self.r_chain[i] = temp_env
                self.posteriors[i] = np.exp(new_posterior)
                if self.posteriors[i] > self.map_posterior:
                    self.map_posterior = self.posteriors[i]
                    self.map_env.set_rewards(temp_env.feature_weights)
                curr_dist = np.linalg.norm(self.gt_reward - temp_env.feature_weights)
                self.closenesses.append(curr_dist)
            else:
                del temp_env
                if self.mcmc_reject:
                    self.r_chain[i] = copy.deepcopy(self.env)
                    curr_dist = np.linalg.norm(self.gt_reward - self.r_chain[i].feature_weights)
                    self.closenesses.append(curr_dist)
                else:
                    i -= 1
            i += 1
    

    def calc_posterior(self, env):
        posterior = 0
        prior = 0
        posterior += prior
        weights = env.feature_weights
        for i in range(len(self.preferences)):
            worse_idx, better_idx = self.preferences[i]
            fcounts_worse = self.feature_counts[worse_idx]
            fcounts_better = self.feature_counts[better_idx]
            worse_return = np.dot(weights, fcounts_worse)
            better_return = np.dot(weights, fcounts_better)
            z = (self.alpha * better_return, self.alpha * worse_return)
            likelihood = self.alpha * better_return - logsumexp(z)
            posterior += likelihood
        return posterior
    

    def get_evolution(self):
        return self.closenesses
    

    def calc_feature_counts(self, trajectory):
        feature_counts = [0 for _ in range(self.init_env.num_features)]
        for i in range(len(trajectory)):
            state, _ = trajectory[i]
            features = self.init_env.state_features[state]
            for j in range(self.init_env.num_features):
                feature_counts[j] += self.gamma**i * features[j]
        return feature_counts
    

    def add_trajectories(self, traj_demonstrations):
        for trajectory in traj_demonstrations:
            fcounts = self.calc_feature_counts(trajectory)
            self.feature_counts.append(fcounts)
    

    def add_preferences(self, prefs):
        for pref in prefs:
            self.preferences.append(pref)
