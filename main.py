import os
import torch
import pickle
import matplotlib
import seaborn as sns
from datetime import datetime
from torch.nn import functional as F
from active_learning import uncertainty_sampling, thompson_sampling, random_sampling
from environments import SineEnv
from experiments import train, predict
from models import Model
from optimizers import SGLD
from parameters import Parameters
from utils import init_uniform, set_seed, plot_uncert, lineplot
sns.set()

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'


def actor(type, models, unobs_x):
    if type == "random":
        return random_sampling(unobs_x)
    elif type == "uncertainty":
        return uncertainty_sampling(models, unobs_x)
    elif type == "thompson":
        return thompson_sampling(models, unobs_x)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # Setting up parameters
    parms = Parameters()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    plot_dir = "plots/" + parms.algo + \
        datetime.now().strftime("_%Y-%m-%d_%H-%M-%S") + "/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print("Algorithm:", parms.algo)

    # Initialize model
    model = Model(out_features=1).type(dtype)
    model.apply(init_uniform)

    # Prepare data
    env = SineEnv(parms, dtype)

    # Run experiments
    test_losses_by_seed = []
    for seed in parms.seeds:
        print(f'==================== Seed {seed} ====================')
        set_seed(seed)
        env.reset()

        # Create plot directory for each seed
        plot_dir_seed = plot_dir + "seed_{}".format(seed) + "/"
        if not os.path.exists(plot_dir_seed):
            os.makedirs(plot_dir_seed)

        # Burn-in iterations
        _, models = train(model=model,
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
                          weight_decay=parms.weight_decay)

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
            save_file=os.path.join(
                plot_dir, "seed_{}_initial.png".format(seed))
        )

        # Active learning
        test_losses = []
        num_iters = parms.al_iter  # env.unobs_x.shape[0]

        for it in range(num_iters):
            print("Iteration:", it+1)

            action = actor(type=parms.algo, models=models, unobs_x=env.unobs_x)

            env.step(action)

            _, models, = train(model=model,
                               loss_fct=parms.loss_fn,
                               x=env.x,
                               y=env.y,
                               batch_size=parms.batch_size,
                               optim=SGLD,
                               gamma=parms.gamma,
                               burnin_iter=parms.loop_epochs,
                               mcmc_iter=parms.mcmc_epochs,
                               lr=parms.lr,
                               num_epochs=parms.inital_epochs,
                               weight_decay=parms.weight_decay)

            y_pred_mean, ale, epi, uncert = predict(models=models, x=x_test)
            test_loss = F.mse_loss(y_pred_mean, y_test.view(-1)).item()
            test_losses.append(test_loss)
            print("Test loss:", test_loss)

            plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean, x_train=env.x, y_train=env.y,
                        ale=ale, epi=epi, uncert=uncert, save_file=os.path.join(
                            plot_dir_seed, "iter_{}.png".format(it+1)))

            # Plot test loss
            if (it + 1) % parms.test_interval == 0:
                lineplot(xs=list(range(1, it+2, parms.test_interval)),
                         ys=test_losses,
                         xlabel="Iteration",
                         ylabel="MSE Loss",
                         ylim=50,
                         save_file=os.path.join(
                                plot_dir_seed, "test_loss.png")
                         )

        test_losses_by_seed.append(test_losses)

    pickle.dump(test_losses_by_seed, open(
        os.path.join(plot_dir, "test_losses_by_seed.pkl"), "wb"))
