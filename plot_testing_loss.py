import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

RESULTS_DIR = './plots'
list_exp_folders = ['random_2023-03-09_17-22-07',
                    'thompson_2023-03-09_17-23-48', 'uncertainty_2023-03-09_17-22-42']
list_algo = ['Random', 'Uncertainty Sampling', 'Thompson Sampling']
num_iter = 3
test_interval = 1
ylim = 50

if __name__ == "__main__":
    plt.figure()

    for algo, exp_folder in zip(list_algo, list_exp_folders):
        test_losses_by_seed = pickle.load(
            open(os.path.join(RESULTS_DIR, exp_folder, 'test_losses_by_seed.pkl'), 'rb'))

        test_losses_by_seed = np.array(test_losses_by_seed)
        mean_test_losses = np.mean(test_losses_by_seed, axis=0)
        min_test_losses = np.min(test_losses_by_seed, axis=0)
        max_test_losses = np.max(test_losses_by_seed, axis=0)

        plt.plot(np.arange(1, num_iter//test_interval + 1)*test_interval,
                 mean_test_losses,
                 label=algo)

        plt.fill_between(np.arange(1, num_iter//test_interval + 1)*test_interval,
                         min_test_losses,
                         max_test_losses,
                         alpha=.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.ylim(0, ylim)
    plt.savefig(os.path.join(RESULTS_DIR, "test_loss_overall.png"), dpi=300)
