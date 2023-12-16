from mdp_utils import *
import numpy as np
import copy


class BIRL:
    def __init__(self, env, demos, beta, epsilon = 0.0001):
        self.env = copy.deepcopy(env)
        self.gt_reward = self.env.feature_weights
        self.demonstrations = demos
        self.epsilon = epsilon
        self.beta = beta
        self.closenesses = []

        #check to see if FeatureMDP or just plain MDP
        if hasattr(self.env, 'feature_weights'):
            self.num_mcmc_dims = len(self.env.feature_weights)
        else:
            self.num_mcmc_dims = self.env.num_states
    

    def reward_prior(self, hyp_reward):
        return 1
    
    
    def calc_ll(self, hyp_reward):
        self.env.set_rewards(hyp_reward)
        q_values = calculate_q_values(self.env, epsilon = self.epsilon)
        #calculate the log likelihood of the reward hypothesis given the demonstrations
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        for s, a in self.demonstrations:
            Z_exponents = self.beta * q_values[s]
            log_sum += self.beta * q_values[s][a] - logsumexp(Z_exponents)
        return log_sum


    def generate_proposal(self, old_sol, stdev, normalize):
        proposal_r = old_sol + stdev * np.random.randn(len(old_sol)) 
        if normalize:
            proposal_r /= np.linalg.norm(proposal_r)
        return proposal_r


    def initial_solution(self):
        return np.zeros(self.num_mcmc_dims)
    

    def run_mcmc(self, max_samples, stepsize, normalize = True, adaptive = True):
        stdev = stepsize  # initial guess for standard deviation, doesn't matter too much
        accept_cnt = 0  # keep track of how often MCMC accepts

        # For adaptive step sizing
        accept_target = 0.44 # ideally around 40% of the steps accept; if accept count is too high, increase stdev, if too low reduce
        horizon = max_samples // 500 # how often to update stdev based on sliding window avg with window size horizon
        learning_rate = 0.05 # how much to update the stdev
        accept_cnt_list = [] # list of accepts and rejects for sliding window avg
        stdev_list = [] # list of standard deviations for debugging purposes
        accept_prob_list = [] # true cumulative accept probs for debugging purposes
        all_lls = [] # all the likelihoods

        self.chain = np.zeros((max_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        self.likelihoods = [0 for _ in range(max_samples)]
        cur_sol = self.initial_solution() #initial guess for MCMC
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll
        map_sol = cur_sol

        i = 0
        curr_dist = np.inf
        while i < max_samples and curr_dist > self.epsilon:
            # sample from proposal distribution
            prop_sol = self.generate_proposal(cur_sol, stdev, normalize)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
            all_lls.append(prop_ll)
            if prop_ll > cur_ll:
                # accept
                self.chain[i, :] = prop_sol
                self.likelihoods[i] = prop_ll
                accept_cnt += 1
                if adaptive:
                    accept_cnt_list.append(1)
                cur_sol = prop_sol
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_sol = prop_sol
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain[i, :] = prop_sol
                    self.likelihoods[i] = prop_ll
                    accept_cnt += 1
                    if adaptive:
                        accept_cnt_list.append(1)
                    cur_sol = prop_sol
                    cur_ll = prop_ll
                else:
                    # reject
                    self.chain[i, :] = cur_sol
                    self.likelihoods[i] = cur_ll
                    if adaptive:
                        accept_cnt_list.append(0)
            curr_dist = np.linalg.norm(self.gt_reward - cur_sol)
            self.closenesses.append(curr_dist)
            # Check for step size adaptation
            if adaptive:
                if len(accept_cnt_list) >= horizon:
                    accept_est = np.sum(accept_cnt_list[-horizon:]) / horizon
                    stdev = max(0.00001, stdev + learning_rate/np.sqrt(i + 1) * (accept_est - accept_target))
                stdev_list.append(stdev)
                accept_prob_list.append(accept_cnt / len(self.chain))
            i += 1
        self.accept_rate = accept_cnt / len(self.closenesses)
        self.map_sol = map_sol
        

    def get_map_solution(self):
        return self.map_sol
    

    def get_solution(self):
        for i in range(4999, -1, -1):
            if self.chain[i, :] is not None:
                return self.chain[i, :]
    
    
    def get_evolution(self):
        return self.closenesses


    def get_mean_solution(self, burn_frac = 0.1, skip_rate = 2):
        burn_indx = int(len(self.chain) * burn_frac)
        mean_r = np.mean(self.chain[burn_indx::skip_rate], axis = 0)
        return mean_r
