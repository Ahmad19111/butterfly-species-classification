"""Seeds für reproduzierbare Läufe."""
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Jedem Worker einen reproduzierbaren Seed geben.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
