import numpy as np
import torch


def mixup_data(x, y, border=None, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    if border is not None:
        mixed_border = lam * border + (1 - lam) * border[index, :]
    else:
        mixed_border = None

    y_a, y_b = y, y[index]
    return mixed_x, mixed_border, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
