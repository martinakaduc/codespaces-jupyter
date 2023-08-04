import torch.nn.functional as F


class Parameters:
    def __init__(self, args, all=False) -> None:
        self.exp_id = f"exp_{args.exp_id:03d}"
        self.loss_fn = F.mse_loss
        self.lr = 1e-4
        self.weight_decay = 0.1
        self.gamma = 1.0

        if not all:
            self.algo = args.algo  # "random" or "uncertainty" or "thompson"
            self.seeds = args.seeds
            self.allow_replacement = args.allow_replacement

        self.train_size = 100
        self.test_size = 400
        self.batch_size = 32
        self.test_interval = 1
        self.al_iter =  100
        
        # Model
        self.input_dim = 1
        self.output_dim = 1
        self.hidden_dim = 32
        
        # SGLD
        self.burnin_epochs = 2000  # 2000
        self.finetune_epochs = 100  # 500
        self.mcmc_epochs = 10  # 100
        self.num_models = 10
        self.inital_epochs = self.burnin_epochs + self.mcmc_epochs * self.num_models
        self.loop_epochs = self.finetune_epochs + self.mcmc_epochs * self.num_models
        

        if args.env == "Sine":
            self.scale = 10
            self.sigma = 2.0
            self.min_x = -2
            self.max_x = 2
        else:
            raise NotImplementedError
