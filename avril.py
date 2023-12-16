from mdp_utils import *
import numpy as np
import copy
import torch
from torch import nn, optim
import torch.nn.functional as F


class AVRIL:
    def __init__(self, env, inputs, targets, state_dim, action_dim, num_encoder_layers = 3, encoder_units = 64, num_qnet_layers = 3, qnet_units = 64, lr = 1e-4, epsilon = 0.0001):
        self.env = copy.deepcopy(env)
        self.gt_reward = env.feature_weights
        self.inputs = inputs
        self.targets = targets
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_qnet_layers = num_qnet_layers
        self.encoder_units = encoder_units
        self.qnet_units = qnet_units
        self.lr = lr
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.closenesses = []
        
        encoder_layers = [nn.Linear(state_dim, encoder_units), nn.ELU()]
        for _ in range(num_encoder_layers - 2):
            encoder_layers.extend([nn.Linear(encoder_units, encoder_units), nn.ELU()])
        encoder_layers.append(nn.Linear(encoder_units, 2))
        self.encoder = nn.Sequential(*encoder_layers).float().to(self.device)

        q_layers = [nn.Linear(state_dim, qnet_units), nn.ELU()]
        for _ in range(num_qnet_layers - 2):
            q_layers.extend([nn.Linear(qnet_units, qnet_units), nn.ELU()])
        q_layers.append(nn.Linear(qnet_units, action_dim))
        self.q_net = nn.Sequential(*q_layers).float().to(self.device)

        params = list(self.encoder.parameters()) + list(self.q_net.parameters())
        self.optimizer = optim.AdamW(params, lr = self.lr)
    

    def reward(self):
        state_features = np.eye(self.env.num_features)
        rew = []
        for sf in state_features:
            if not isinstance(sf, torch.Tensor):
                sf = torch.from_numpy(sf).float().to(self.device)
            with torch.no_grad():
                out = self.encoder(sf)[0].item()
                rew.append(out)
        rew = np.array(rew)
        rew = rew / np.linalg.norm(rew)
        return rew
    

    def elbo(self, states, next_states, actions, next_actions):
        assert len(states[0]) == self.env.num_features, "Must reshape states"
        assert len(next_states[0]) == self.env.num_features, "Must reshape next_states"
        assert len(states) == len(next_states), "len(states) != len(next_states)"

        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        next_actions = torch.from_numpy(next_actions).long().to(self.device)

        q_values = self.q_net(states)
        q_values_a = torch.gather(q_values, 1, actions.unsqueeze(0)).squeeze(0).reshape(len(states)).to(self.device)
        q_values_next = self.q_net(next_states)
        q_values_next_a = torch.gather(q_values_next, 1, next_actions.unsqueeze(0)).reshape(len(next_states)).squeeze(0).to(self.device)

        td = q_values_a - q_values_next_a
        reward = self.encoder(states)
        means = reward[:, 0].reshape(len(states))
        log_sds = reward[:, 1].reshape(len(states))
        
        irl_loss = -torch.distributions.Normal(means, torch.exp(log_sds.detach())).log_prob(td).mean()
        kl = kl_gaussian(means, torch.exp(log_sds.detach()) ** 2).mean()
        pred = F.log_softmax(q_values, dim = -1)
        neg_ll = -torch.gather(pred, 1, actions.unsqueeze(0)).squeeze(0).to(self.device).mean()
        return neg_ll + kl + irl_loss
    

    def update(self, states, next_states, actions, next_actions):
        self.optimizer.zero_grad()
        loss = self.elbo(states, next_states, actions, next_actions)
        loss.backward()
        self.optimizer.step()
    
    
    def train(self, max_iters, batch_size = 8):
        states, next_states = self.inputs
        actions, next_actions = self.targets
        input_length = len(states)
        num_batches = np.ceil(input_length / batch_size)
        idx_list = np.array(range(input_length))

        i = 0
        curr_dist = np.inf
        while i < max_iters and curr_dist > self.epsilon:
            if i % num_batches == 0:
                idx_list_shuffle = torch.randperm(len(idx_list))
            idx = int((i % num_batches) * batch_size)
            indices = idx_list_shuffle[idx : idx + batch_size].detach().numpy()
            self.update(np.array(states)[indices],
                        np.array(next_states)[indices],
                        np.array(actions)[indices],
                        np.array(next_actions)[indices])
            curr_dist = np.linalg.norm(self.gt_reward - self.reward())
            self.closenesses.append(curr_dist)
            i += 1


    def get_evolution(self):
        return self.closenesses
