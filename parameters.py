import torch.nn.functional as F


class Parameters:
    def __init__(self) -> None:
        self.loss_fn = F.mse_loss
        self.lr = 1e-4
        self.weight_decay = .1
        self.gamma = 1.

        self.algo = "uncertainty"  # "random" or "uncertainty" or "thompson"
        self.seeds = [2, 3, 5]
        self.batch_size = 32
        self.burnin_epochs = 2000  # 2000
        self.finetune_epochs = 500  # 500
        self.mcmc_epochs = 100  # 100
        self.num_models = 10
        self.inital_epochs = self.burnin_epochs + self.mcmc_epochs * self.num_models
        self.loop_epochs = self.finetune_epochs + self.mcmc_epochs * self.num_models
        self.test_interval = 1
        self.al_iter = 100  # 100

        self.train_size = 100
        self.test_size = 400
        self.sigma = 2.0
        self.min_x = -2
        self.max_x = 2
