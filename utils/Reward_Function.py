import torch


def difference(target_value, value, sigma):
    return ((target_value - value) / sigma) ** 2


def abs_sum(target_value, value):
    if value.dim() > 1:
        return -torch.sum(torch.abs(target_value - value), dim=1)
    else:
        return -torch.abs(target_value - value)


def exp_sum(target_value, value, sigma,dim=2):
    if dim>1:
        return torch.exp(-(torch.sum(difference(target_value, value, sigma), dim=1)))
    else:
        return torch.exp(-difference(target_value, value, sigma))



def potential_reward(target_value, current_value, next_value, gamma=1):
    next_exp_reward = abs_sum(target_value, next_value)
    current_exp_reward = abs_sum(target_value, current_value)
    return next_exp_reward - current_exp_reward * gamma


