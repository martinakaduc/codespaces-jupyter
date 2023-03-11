import os
import pickle
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from parameters import Parameters

sns.set()

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'


args = argparse.ArgumentParser()
args.add_argument("--exp_id", type=int, default=0)
args = args.parse_args()

parms = Parameters(args, all=True)
    
RESULTS_DIR = f'./plots/{parms.exp_id}'
list_exp_folders = ['random', 'uncertainty', 'thompson']
list_algo = ['Random', 'Uncertainty Sampling', 'Thompson Sampling']

if __name__ == "__main__":
    plt.figure()

    for algo, exp_folder in zip(list_algo, list_exp_folders):
        test_losses_by_seed = pickle.load(
            open(os.path.join(RESULTS_DIR, exp_folder, 'test_losses_by_seed.pkl'), 'rb'))

        test_losses_by_seed = np.array(list(test_losses_by_seed.values()))
        mean_test_losses = np.mean(test_losses_by_seed, axis=0)
        min_test_losses = np.min(test_losses_by_seed, axis=0)
        max_test_losses = np.max(test_losses_by_seed, axis=0)

        plt.plot(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                 mean_test_losses,
                 label=algo)

        plt.fill_between(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                         min_test_losses,
                         max_test_losses,
                         alpha=.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.ylim(0, 50)
    plt.savefig(os.path.join(RESULTS_DIR, "test_loss_overall.png"), dpi=300)
    plt.close()
    
    
    
    plt.figure()

    for algo, exp_folder in zip(list_algo, list_exp_folders):
        test_losses_by_seed = pickle.load(
            open(os.path.join(RESULTS_DIR, exp_folder, 'average_observed_y_by_seed.pkl'), 'rb'))

        test_losses_by_seed = np.array(list(test_losses_by_seed.values()))
        mean_test_losses = np.mean(test_losses_by_seed, axis=0)
        min_test_losses = np.min(test_losses_by_seed, axis=0)
        max_test_losses = np.max(test_losses_by_seed, axis=0)

        plt.plot(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                 mean_test_losses,
                 label=algo)

        plt.fill_between(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                         min_test_losses,
                         max_test_losses,
                         alpha=.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Average y')
    plt.ylim(0, 15)
    plt.savefig(os.path.join(RESULTS_DIR, "average_y.png"), dpi=300)
    plt.close()
    
    
    
    plt.figure()

    for algo, exp_folder in zip(list_algo, list_exp_folders):
        test_losses_by_seed = pickle.load(
            open(os.path.join(RESULTS_DIR, exp_folder, 'percentage_observed_top1_by_seed.pkl'), 'rb'))

        test_losses_by_seed = np.array(list(test_losses_by_seed.values()))
        mean_test_losses = np.mean(test_losses_by_seed, axis=0)
        min_test_losses = np.min(test_losses_by_seed, axis=0)
        max_test_losses = np.max(test_losses_by_seed, axis=0)

        plt.plot(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                 mean_test_losses,
                 label=algo)

        plt.fill_between(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                         min_test_losses,
                         max_test_losses,
                         alpha=.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Top-1 percentage')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(RESULTS_DIR, "top_1_percentage.png"), dpi=300)
    plt.close()
    
    
    
    plt.figure()

    for algo, exp_folder in zip(list_algo, list_exp_folders):
        test_losses_by_seed = pickle.load(
            open(os.path.join(RESULTS_DIR, exp_folder, 'percentage_observed_top5_by_seed.pkl'), 'rb'))

        test_losses_by_seed = np.array(list(test_losses_by_seed.values()))
        mean_test_losses = np.mean(test_losses_by_seed, axis=0)
        min_test_losses = np.min(test_losses_by_seed, axis=0)
        max_test_losses = np.max(test_losses_by_seed, axis=0)

        plt.plot(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                 mean_test_losses,
                 label=algo)

        plt.fill_between(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                         min_test_losses,
                         max_test_losses,
                         alpha=.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Top-5 percentage')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(RESULTS_DIR, "top_5_percentage.png"), dpi=300)
    plt.close()
    
    
    
    plt.figure()

    for algo, exp_folder in zip(list_algo, list_exp_folders):
        test_losses_by_seed = pickle.load(
            open(os.path.join(RESULTS_DIR, exp_folder, 'percentage_observed_top10_by_seed.pkl'), 'rb'))

        test_losses_by_seed = np.array(list(test_losses_by_seed.values()))
        mean_test_losses = np.mean(test_losses_by_seed, axis=0)
        min_test_losses = np.min(test_losses_by_seed, axis=0)
        max_test_losses = np.max(test_losses_by_seed, axis=0)

        plt.plot(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                 mean_test_losses,
                 label=algo)

        plt.fill_between(np.arange(1, parms.al_iter//parms.test_interval + 1)*parms.test_interval,
                         min_test_losses,
                         max_test_losses,
                         alpha=.2)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Top-10 percentage')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(RESULTS_DIR, "top_10_percentage.png"), dpi=300)
    plt.close()
