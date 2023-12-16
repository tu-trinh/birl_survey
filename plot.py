import matplotlib.pyplot as plt
import numpy as np
import pickle


ALL_METHODS = ["birl", "brex", "avril"]
ALL_NUM_FEATURES = [2, 4, 8]
ALL_PERC_DEMOS = [25, 50, 75, 100]
COLORS = {"birl": "red", "brex": "blue", "avril": "green"}


def get_method_logs(method, feat, demo):
    evolutions = []
    for seed in range(10):
        file_name = f"./logs/{method}/feat{feat}_demo{demo}_seed{seed}.pkl"
        with open(file_name, "rb") as f:
            evolution = pickle.load(f)
            if not isinstance(evolution, list):
                evolutions.append(evolution.tolist())
            else:
                evolutions.append(evolution)
    return evolutions


def calc_mean_and_std(evolutions):
    sums = {}
    sums_of_squares = {}
    counts = {}
    for evolution in evolutions:
        for i, dist in enumerate(evolution):
            if i not in sums:
                sums[i] = 0
                sums_of_squares[i] = 0
                counts[i] = 0
            sums[i] += dist
            sums_of_squares[i] += dist ** 2
            counts[i] += 1
    means = []
    stds = []
    for i in sorted(sums.keys()):
        mean = sums[i] / counts[i]
        std = ((sums_of_squares[i] / counts[i]) - mean ** 2) ** 0.5
        means.append(mean)
        stds.append(std)
    assert len(means) == len(stds), "you fucked up"
    return (means, stds)


def sliding_window_average(data, window_size = 20):
    averaged_data = [np.mean(data[max(i - window_size + 1, 0) : i + 1]) for i in range(len(data))]
    return averaged_data


def plot_data(data, feat, demo):
    plt.figure()
    for method in ALL_METHODS:
        mean, std = data[method]
        mean_avg = sliding_window_average(mean)
        std_avg = sliding_window_average(std)
        x = range(len(mean_avg))
        plt.plot(x, mean_avg, label = method.upper(), color = COLORS[method])
        # plt.fill_between(x, [m - s for m, s in zip(mean_avg, std_avg)], [m + s for m, s in zip(mean_avg, std_avg)],
        #                  color = COLORS[method], alpha = 0.3)
    plt.xlabel("Iteration")
    plt.ylabel("Distance to Ground Truth Reward")
    plt.title(f"{feat} Features, {demo}% States as Demos")
    plt.legend()
    plt.savefig(f"./plots/conv_feat{feat}_demo{demo}.png")


def plot_env(gt_reward, state_features):
    reward_grid = []
    for sf in state_features:
        reward_idx = np.where(sf == 1)[0][0]
        reward_grid.append(gt_reward[reward_idx])
    


def main():
    for num_features in ALL_NUM_FEATURES:
        for perc_demo in ALL_PERC_DEMOS:
            evolutions = {}
            data = {}
            for method in ALL_METHODS:
                evolutions[method] = get_method_logs(method, num_features, perc_demo)
                data[method] = calc_mean_and_std(evolutions[method])
            plot_data(data, num_features, perc_demo)


def alt():
    for num_features in [4]:
        for perc_demo in [100]:
            for seed in range(10):
                file_name = f"./logs/brex/feat{num_features}_demo{perc_demo}_seed{seed}.pkl"
                with open(file_name, "rb") as f:
                    env_data = pickle.load(f)
                    gt_reward = env_data[gt_reward]
                    state_features = env_data[state_features]
                
                method_rewards = {}
                for method in ALL_METHODS:
                    file_name = file_name = f"./logs/{method}/feat{feat}_demo{demo}_seed{seed}.pkl"
                    with open(file_name, "rb") as f:
                        final_rew = pickle.load(f)[-1]
                        # TODO: FUCKKKKK
                    plot_env(gt_reward, state_features)


if __name__ == "__main__":
    # main()
    alt()