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
    #print(f'D(x): {torch.mean(pos):.5f}, D(G(z)): {torch.mean(neg):.5f}, dloss: {d_loss:.4f}')
    return d_loss

def hinge_loss_g(neg):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    g_loss = -torch.mean(neg)
    #print(f'D(G(z)): {torch.mean(neg):.5f}')
    return g_loss


def wasserstein_loss_d(pos, neg):
    """ maffe implementation """
    d_loss = 0.5 * torch.mean(neg) - 0.5 * torch.mean(pos)
    return d_loss

def wasserstein_loss_g(neg):
    """ maffe implementation """
    g_loss = -torch.mean(neg)
    return g_loss

def loss_l1(x_in, x_out, penalty=1.0):
    """ maffe implementation: asymmetrical l1 loss that penalizes more if out>in
    Note that if penalty=1.0 this is just the original l1 loss
    x_in: input tensor
    x_out: output tensor. Both are torch tensors (N, 3, 256, 256) """
    diff = x_out - x_in # (N, 3, 256, 256)
    loss = torch.where(diff > 0, penalty*torch.abs(diff), 1.0*torch.abs(diff)) # (N, 3, 256, 256)
    return torch.mean(loss)

def loss_l1_l2(x_in, x_out):
    """
    x_in: input tensor
    x_out: output tensor. Both are torch tensors (N, 3, 256, 256) """
    diff = x_out - x_in # (N, 3, 256, 256)
    loss = torch.where(diff > 0, torch.pow(diff, 2), torch.abs(diff)) # (N, 3, 256, 256)
    return torch.mean(loss)

def loss_power_law(dem, bed, mask, c, gamma, mins, maxs, ris_lon, ris_lat):
    """
    maffe implementation
    c: scaling parameter [ca. 0.03]
    gamma: exponent [ca. 1.4]
    dem (N, 256, 256)
    bed (N, 256, 256)
    mask (1, 256, 256)
    V_th = c * A^gamma
    note: for each item ris_lon and ris_lat are different.
    """
    areas = torch.sum(mask) # scalar
    vol_th = c * torch.pow(areas*ris_lon*ris_lat, gamma)  # (N,)

    # thickness = dem - bed # (N, 256, 256)
    thickness = torch.sum(dem - bed, dim=(1, 2))   # (N,)
    vol_exp = 0.5 * (maxs - mins) * thickness * ris_lon * ris_lat # (N,)

    loss = torch.abs(vol_th - vol_exp) # (N,)
    #print('scaling loss', loss*100/vol_th)

    return torch.mean(loss)