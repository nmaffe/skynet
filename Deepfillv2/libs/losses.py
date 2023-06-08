import torch

def ls_loss_d(pos, neg, value=1.):
    """
    gan with least-square loss
    """
    l2_pos = torch.mean((pos-value)**2)
    l2_neg = torch.mean(neg**2)
    d_loss = 0.5*l2_pos + 0.5*l2_neg 
    return d_loss

def ls_loss_g(neg, value=1.):    
    """
    gan with least-square loss
    """
    g_loss = torch.mean((neg-value)**2)
    return g_loss

def hinge_loss_d(pos, neg):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    hinge_pos = torch.mean(torch.relu(1-pos))
    hinge_neg = torch.mean(torch.relu(1+neg))
    d_loss = 0.5*hinge_pos + 0.5*hinge_neg   
    return d_loss

def hinge_loss_g(neg):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    g_loss = -torch.mean(neg)
    return g_loss


def wesserstein_loss_d(pos, neg):
    """ maffe implementation """
    d_loss = 0.5 * torch.mean(neg) - 0.5 * torch.mean(pos)
    return d_loss

def wesserstein_loss_g(neg):
    """ maffe implementation """
    g_loss = -torch.mean(neg)
    return g_loss

def loss_l1(x_in, x_out, penalty=1.0):
    """ maffe implementation: asymmetrical l1 loss that penalizes more if out>in
    Note that if penalty=1.0 this is just the original l1 loss
    x_in: input tensor
    x_out: output tensor. Both are torch tensors (batch_size, 3, 256, 256) """
    diff = x_out - x_in
    loss = torch.where(diff > 0, penalty*torch.abs(diff), 1.0*torch.abs(diff))
    return torch.mean(loss)