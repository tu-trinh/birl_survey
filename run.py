from mdp import FeatureMDP
from mdp_utils import *
from birl import BIRL
from brex import BREX
from avril import AVRIL

import numpy as np
import random
import argparse
import time
from datetime import timedelta
import pickle


MDP_SIZE = 8
ALL_NUM_FEATURES = [4]
PERCENT_DEMOS = [1]
NUM_ITERS = 1
BETA = 10
TOTAL_ROUNDS = len(ALL_NUM_FEATURES) * len(PERCENT_DEMOS) * NUM_ITERS
MAX_SAMPLES = 5000
STEP_SIZE = 0.5


def get_mdp(num_features):
    gt_reward = np.concatenate((np.random.uniform(low = -1, high = 0, size = num_features - 1),
                                np.random.random()), axis = None)
    gt_reward =  gt_reward / np.linalg.norm(gt_reward)

    feature_vectors = np.eye(num_features)
    state_features = feature_vectors[np.random.choice(range(len(feature_vectors) - 1), size = MDP_SIZE**2)]
    goal_idx = np.random.randint(0, MDP_SIZE**2)
    state_features[goal_idx] = feature_vectors[-1]

    mdp = FeatureMDP(MDP_SIZE, MDP_SIZE, 4, gt_reward, state_features)
    return mdp


def format_time(secs):
    td = timedelta(seconds = secs)
    return str(td)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", required = True, type = str)
    parser.add_argument("--seed", "-s", required = True, type = int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    demo_order = list(range(MDP_SIZE ** 2))
    random.shuffle(demo_order)

    combo_num = 1
    for num_features in ALL_NUM_FEATURES:
        for perc_demo in PERCENT_DEMOS:
            for i in range(NUM_ITERS):
                start = time.time()
                mdp = get_mdp(num_features)
                true_q_values = calculate_q_values(mdp)
                allowed_demos = demo_order[:int(MDP_SIZE**2 * perc_demo)]
                trajectories = [generate_optimal_demo(mdp, init_state, true_q_values) for init_state in allowed_demos]

                if args.method == "birl":
                    demos = [traj[0] for traj in trajectories]
                    birl = BIRL(mdp, demos, beta = BETA)
                    birl.run_mcmc(MAX_SAMPLES, STEP_SIZE)
                    evolution = birl.get_evolution()
                    sol = birl.get_solution()
                
                elif args.method == "brex":
                    traj_returns = [calculate_trajectory_return(mdp, traj) for traj in trajectories]
                    prefs = []
                    for i in range(len(trajectories)):
                        for j in range(i + 1, len(trajectories)):
                            if traj_returns[i] > traj_returns[j]:
                                prefs.append((j, i))
                            elif traj_returns[j] > traj_returns[i]:
                                prefs.append((i, j))
                    brex = BREX(mdp, -1, 1, MAX_SAMPLES, STEP_SIZE, 0.95)
                    brex.add_trajectories(trajectories)
                    brex.add_preferences(prefs)
                    brex.run()
                    evolution = brex.get_evolution()
                    sol = brex.get_solution()
                
                elif args.method == "avril":
                    sf = mdp.state_features
                    states = [sf[s] for traj in trajectories for s, _ in traj[:-1]]
                    next_states = [sf[s] for traj in trajectories for s, _ in traj[1:]]
                    actions = [a for traj in trajectories for _, a in traj[:-1]]
                    next_actions = [a for traj in trajectories for _, a in traj[1:]]
                    inputs = (states, next_states)
                    targets = (actions, next_actions)
                    avril = AVRIL(mdp, inputs, targets, num_features, 4)
                    avril.train(MAX_SAMPLES)
                    evolution = avril.get_evolution()
                    sol = avril.get_solution()
                
                end = time.time()
                print(f"Finished combo {combo_num} of {TOTAL_ROUNDS}, time: {format_time(end - start)}")
                combo_num += 1

                # log_path = f"./logs/{args.method}/feat{num_features}_demo{int(100 * perc_demo)}_seed{args.seed}.pkl"
                # with open(log_path, "wb") as f:
                #     pickle.dump(evolution, f)
                # data = {
                #     "gt_reward": mdp.feature_weights,
                #     "state_features": mdp.state_features
                # }
                # log_path = f"./logs/{args.method}/feat{num_features}_demo{int(100 * perc_demo)}_seed{args.seed}_env.pkl"
                # with open(log_path, "wb") as f:
                #     pickle.dump(data, f)
                log_path = f"./logs/{args.method}/feat{num_features}_demo{int(100 * perc_demo)}_seed{args.seed}_rew.pkl"
                with open(log_path, "wb") as f:
                    pickle.dump(sol, f)
