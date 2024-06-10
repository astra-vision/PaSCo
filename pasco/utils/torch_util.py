import numpy as np
import torch
import random
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)

def set_random_seed(seed):
    """Set random seed for numpy, torch, and python random.
    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    print("set random seed to {}".format(seed))

def enable_dropout(model):
    """Enable dropout for a model."""
    print("enable dropout")
    for m in model.modules():
        if "dropout" in m.__class__.__name__.lower():
            print(m.__class__.__name__)
            m.train()


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]

class WarmupCosine:
    def __init__(self, warmup_end, max_iter, factor_min):
        self.max_iter = max_iter
        self.warmup_end = warmup_end
        self.factor_min = factor_min

    def __call__(self, iter):
        factor = 1.0
        if iter < self.warmup_end:
            factor = min(iter / self.warmup_end, 1.0)
        # else:
        #     iter = iter - self.warmup_end
        #     max_iter = self.max_iter - self.warmup_end
        #     iter = (iter / max_iter) * np.pi
        #     factor = self.factor_min + 0.5 * (1 - self.factor_min) * (np.cos(iter) + 1)
        # return factor
    
        if iter > 60000:
            factor = 0.1
        # if iter > 0000:
        #     factor = 0.01
        return factor

class StepLR:
    def __init__(self, epochs, factors):
        self.epochs = epochs
        self.factors = factors

    def __call__(self, iter):
        factor = 1.0
        for epoch, f in zip(self.epochs, self.factors):
            if iter > epoch:
                factor = f
    
        return factor 