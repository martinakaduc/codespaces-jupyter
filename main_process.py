import os
import torch
import pickle
import argparse
import matplotlib
import seaborn as sns

from torch.nn import functional as F
from active_learning import actor
from environments import SineEnv
from experiments import train, predict
from models import Model
from optimizers import SGLD
from parameters import Parameters
from threading import Thread
from torch.multiprocessing import Process, Manager, set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass

from utils import init_uniform, set_seed, plot_uncert, lineplot

sns.set()

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{bm}"


def run_by_seed(
    seed,
    test_losses_by_seed,
    percentage_observed_top1_by_seed,
    percentage_observed_top5_by_seed,
    percentage_observed_top10_by_seed,
    average_observed_y_by_seed,
    parms,
    dtype,
    plot_dir,
):
    print(f"==================== Seed {seed} ====================")
    set_seed(seed)

    # Initialize model
    model = Model(out_features=1).type(dtype)
    model.apply(init_uniform)

    # Prepare data
    env = SineEnv(parms, dtype)
    env.reset()

    # Create plot directory for each seed
    plot_dir_seed = plot_dir + "seed_{}".format(seed) + "/"
    if not os.path.exists(plot_dir_seed):
        os.makedirs(plot_dir_seed)

    # Burn-in iterations
    _, models = train(
        model=model,
        loss_fct=parms.loss_fn,
        x=env.x,
        y=env.y,
        batch_size=parms.batch_size,
        optim=SGLD,
        gamma=parms.gamma,
        burnin_iter=parms.burnin_epochs,
        mcmc_iter=parms.mcmc_epochs,
        lr=parms.lr,
        num_epochs=parms.inital_epochs,
        weight_decay=parms.weight_decay,
    )

    # Plotting
    x_test, y_test = env.rollout(burnin=False, num_steps=parms.test_size)

    y_pred_mean, ale, epi, uncert = predict(models=models, x=x_test)
    test_loss = F.mse_loss(y_pred_mean, y_test.view(-1)).item()
    print("Test loss:", test_loss)

    plot_uncert(
        x_test=x_test,
        y_pred_mean=y_pred_mean,
        x_train=env.x,
        y_train=env.y,
        ale=ale,
        epi=epi,
        uncert=uncert,
        save_file=os.path.join(plot_dir, "seed_{}_initial.png".format(seed)),
    )

    # Active learning
    test_losses = []
    percentage_observed_top1 = []
    percentage_observed_top5 = []
    percentage_observed_top10 = []
    average_observed_y = []
    num_iters = parms.al_iter  # env.unobs_x.shape[0]

    for it in range(num_iters):
        print("Iteration:", it + 1)

        action = actor(type=parms.algo, models=models, unobs_x=env.unobs_x)

        env.step(action)

        (
            _,
            models,
        ) = train(
            model=model,
            loss_fct=parms.loss_fn,
            x=env.x,
            y=env.y,
            batch_size=parms.batch_size,
            optim=SGLD,
            gamma=parms.gamma,
            burnin_iter=parms.finetune_epochs,
            mcmc_iter=parms.mcmc_epochs,
            lr=parms.lr,
            num_epochs=parms.loop_epochs,
            weight_decay=parms.weight_decay,
        )

        y_pred_mean, ale, epi, uncert = predict(models=models, x=x_test)

        # Record percentage observed
        percentage_observed_top1.append(env.get_percentage_observed_topK(k=1))
        percentage_observed_top5.append(env.get_percentage_observed_topK(k=5))
        percentage_observed_top10.append(env.get_percentage_observed_topK(k=10))

        # Record average observed y
        average_observed_y.append(env.get_average_observed_y())

        # Plot uncertainty
        plot_uncert(
            x_test=x_test,
            y_pred_mean=y_pred_mean,
            x_train=env.x,
            y_train=env.y,
            ale=ale,
            epi=epi,
            uncert=uncert,
            save_file=os.path.join(plot_dir_seed, "iter_{}.png".format(it + 1)),
        )

        # Plot test loss
        if (it + 1) % parms.test_interval == 0:
            test_loss = F.mse_loss(y_pred_mean, y_test.view(-1)).item()
            test_losses.append(test_loss)
            print("Test loss:", test_loss)

            lineplot(
                xs=list(range(1, it + 2, parms.test_interval)),
                ys=test_losses,
                xlabel="Iteration",
                ylabel="MSE Loss",
                ylim=50,
                save_file=os.path.join(plot_dir_seed, "test_loss.png"),
            )

    test_losses_by_seed[seed] = test_losses
    percentage_observed_top1_by_seed[seed] = percentage_observed_top1
    percentage_observed_top5_by_seed[seed] = percentage_observed_top5
    percentage_observed_top10_by_seed[seed] = percentage_observed_top10
    average_observed_y_by_seed[seed] = average_observed_y


if __name__ == "__main__":
    # Setting up parameters
    args = argparse.ArgumentParser()
    args.add_argument("--algo", type=str, default="random")
    args.add_argument("--exp_id", type=int, default=0)
    args.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args.add_argument("--allow_replacement", action="store_true")
    args = args.parse_args()

    parms = Parameters(args)
    if torch.cuda.is_available():
        print("Using GPU")
        dtype = torch.cuda.FloatTensor
    else:
        print("Using CPU")
        dtype = torch.FloatTensor

    plot_dir = f"plots/{parms.exp_id}/" + parms.algo + "/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print("Algorithm:", parms.algo)

    # Run experiments
    test_losses_by_seed = {}
    percentage_observed_top1_by_seed = {}
    percentage_observed_top5_by_seed = {}
    percentage_observed_top10_by_seed = {}
    average_observed_y_by_seed = {}

    with Manager() as manager:
        test_losses_by_seed = manager.dict()
        percentage_observed_top1_by_seed = manager.dict()
        percentage_observed_top5_by_seed = manager.dict()
        percentage_observed_top10_by_seed = manager.dict()
        average_observed_y_by_seed = manager.dict()

        list_threads = []
        for seed in parms.seeds:
            # run_by_seed(seed,
            #             test_losses_by_seed,
            #             percentage_observed_top1_by_seed,
            #             percentage_observed_top5_by_seed,
            #             percentage_observed_top10_by_seed,
            #             average_observed_y_by_seed
            # )

            t = Process(
                target=run_by_seed,
                args=(
                    seed,
                    test_losses_by_seed,
                    percentage_observed_top1_by_seed,
                    percentage_observed_top5_by_seed,
                    percentage_observed_top10_by_seed,
                    average_observed_y_by_seed,
                    parms,
                    dtype,
                    plot_dir,
                ),
            )
            list_threads.append(t)

        for t in list_threads:
            t.start()

        for t in list_threads:
            t.join()

        test_losses_by_seed = dict(test_losses_by_seed)
        percentage_observed_top1_by_seed = dict(percentage_observed_top1_by_seed)
        percentage_observed_top5_by_seed = dict(percentage_observed_top5_by_seed)
        percentage_observed_top10_by_seed = dict(percentage_observed_top10_by_seed)
        average_observed_y_by_seed = dict(average_observed_y_by_seed)

    # Save results
    pickle.dump(
        test_losses_by_seed,
        open(os.path.join(plot_dir, "test_losses_by_seed.pkl"), "wb"),
    )

    pickle.dump(
        percentage_observed_top1_by_seed,
        open(os.path.join(plot_dir, "percentage_observed_top1_by_seed.pkl"), "wb"),
    )

    pickle.dump(
        percentage_observed_top5_by_seed,
        open(os.path.join(plot_dir, "percentage_observed_top5_by_seed.pkl"), "wb"),
    )

    pickle.dump(
        percentage_observed_top10_by_seed,
        open(os.path.join(plot_dir, "percentage_observed_top10_by_seed.pkl"), "wb"),
    )

    pickle.dump(
        average_observed_y_by_seed,
        open(os.path.join(plot_dir, "average_observed_y_by_seed.pkl"), "wb"),
    )
