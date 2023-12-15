import numpy as np


class MDP:
    def __init__(self, num_rows, num_cols, num_actions, rewards, gamma = 0.95, noise = 0.1):
        self.gamma = gamma
        self.num_states = num_rows * num_cols
        self.num_actions = num_actions  # up:0, down:1, left:2, right:3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.rewards = rewards
        
        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)


    def init_transition_probabilities(self, noise):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

        # going UP
        for s in range(self.num_states):
            # possibility of going foward
            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)
            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise
            # possibility of going right
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise
            # special case top left corner
            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise
        
        # going down
        for s in range(self.num_states):
            # possibility of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][DOWN][s + self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * noise)
            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = noise
            else:
                self.transitions[s][DOWN][s - 1] = noise
            # possibility of going right
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][DOWN][s + 1] = noise
            else:
                self.transitions[s][DOWN][s] = noise
            # checking bottom right corner
            if s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = 1.0 - noise
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][DOWN][s] = 1.0 - noise
        
        # going left
        for s in range(self.num_states):
            # possibility of going left
            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)
            # possibility of going up
            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise
            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise
            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)
            # possibility of going up
            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise
            # possibility of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise
            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

    
    def set_rewards(self, _rewards):
        self.rewards = _rewards

    def set_gamma(self, gamma):
        assert(gamma < 1.0 and gamma > 0.0)
        self.gamma = gamma



class FeatureMDP(MDP):        
    def __init__(self, num_rows, num_cols, num_actions, feature_weights, state_features, gamma = 0.95, noise = 0.1):
        assert(num_rows * num_cols == len(state_features))
        assert(len(state_features[0]) == len(feature_weights))

        #rewards are linear combination of features
        rewards = np.dot(state_features, feature_weights)
        super().__init__(num_rows, num_cols, num_actions, rewards, gamma, noise)

        self.feature_weights = feature_weights
        self.state_features = state_features
        self.num_features = len(feature_weights)

    def set_rewards(self, _feature_weights):
        assert(len(_feature_weights) == len(self.feature_weights))
        self.rewards = np.dot(self.state_features, _feature_weights)
        self.feature_weights = _feature_weights


if __name__ =="__main__":

    '''Here's a simple example of how to use the FeatureMDP class'''

    
    #three features, red (-1), blue (+1), white (0)
    r = [1,0]
    b = [0,1]
    w = [0,0]
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, 
                      w, w]
    feature_weights = [-1.0, 1.0] #red feature has weight -1 and blue feature has weight +1
    gamma = 0.5
    noise = 0.0
    eps = 0.0001
    env = FeatureMDP(2,2,[0],feature_weights, state_features, gamma, noise)
    
    # mdp_utils.value_iteration(env)