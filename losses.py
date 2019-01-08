import torch.nn.functional as F


def nll_loss(y, y_target):
    return F.nll_loss(y, y_target)


def contastive_loss(x, x_sampling, take_pos, margin=1.0):
    distance = (x - x_sampling).norm(p=2, dim=1, keepdim=True)

    take_pos = take_pos.view(-1, 1)
    pos_part = take_pos * distance.pow(2)
    neg_part = (1.0 - take_pos) * F.relu(margin - distance).pow(2)

    loss = 0.5 * (pos_part + neg_part)
    return loss.mean()


def tripet_loss(anchor, pos, neg, margin=1.0):
    # todo: implement it maually
    return F.triplet_margin_loss(anchor, pos, neg, margin=margin).mean()


def binary_cross_entropy(y, y_target):
    return F.binary_cross_entropy(y, y_target)
