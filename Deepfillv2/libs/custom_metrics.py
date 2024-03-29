import torch
import Deepfillv2.libs.misc as misc

def RMSE_MAE(y1,y2,mins,maxs,mask):
    '''
    Code by Niccolo
    Compute the RMSE between two denormalized images in the reconstructed region
    y1 (N, 1, 256, 256)
    y2 (N, 1, 256, 256)
    mins (N, )
    maxs (N, )
    mask (N, 1, 256, 256)
    '''
    y1 = y1.detach()
    y2 = y2.detach()
    mins = mins.detach()
    maxs = maxs.detach()
    mask = mask.detach()

    y1 = misc.pt_to_image_denorm(y1, min=mins, max=maxs).detach()  # (N, 1, 256, 256)
    y2 = misc.pt_to_image_denorm(y2, min=mins, max=maxs).detach()  # (N, 1, 256, 256)

    #mask_size = torch.sum(mask).detach() # GL scalare
    #rmse_GL = torch.sqrt(256 * 256 * torch.mean((y1 - y2) ** 2) / mask_size) # GL scalare

    mask_sizes = torch.sum(mask, dim=(1,2,3)).detach()      # (N,)
    sum_squared_diff = torch.sum((y1-y2)**2, dim=(1,2,3))   # (N,)
    sum_absolute_diff = torch.sum(torch.abs(y1-y2), dim=(1,2,3)) # (N,)

    rmse = torch.sqrt(sum_squared_diff/mask_sizes)          # (N,)
    mae = torch.sqrt(sum_absolute_diff/mask_sizes)          # (N,)

    return torch.mean(rmse), torch.mean(mae)

def MRE(y1,y2,mins,maxs,mask):
    '''
    code by Gianluca Lagnese
    Compute the Mean Reconstruction Error (MRE) between two denormalized images
    in the reconstructed region
    '''

    mask_size=torch.sum(mask).detach()
    x1=misc.pt_to_image_denorm(y1,mins,maxs) # ha senso cosi com'è implementato?
    x2=misc.pt_to_image_denorm(y2,mins,maxs)
    

    return 256*256*torch.mean(x1-x2)/mask_size

#def mask_dilation(mask,direction=None):
#
#def RMSE_slopes(y1,y2,ris_lon,ris_lat,mins,maxs,mask):
#    '''
#    Compute the RMSE between two denormalized images in the reconstructed region
#    '''
#
#    mask_size=torch.sum(mask).detach()
##    x1=misc.pt_to_image(y1)
#
##    x2=misc.pt_to_image(y2)
#    x1=misc.pt_to_image_denorm(y1,mins,maxs) # ha senso cosi com'è implementato?
#    x2=misc.pt_to_image_denorm(y2,mins,maxs)
#    
##    x1=(maxs-mins)*x1+mins
##    x2=(maxs-mins)*x2+mins
#
#    return torch.sqrt(256*256*torch.mean((x1-x2)**2)/mask_size)
#
#def curvature(Z,ris_lat,ris_lon):
#    '''
#    Compute the curvature of a discrete surface defined as Z=Z(X,Y). See for instance:
#    https://en.wikipedia.org/wiki/Differential_geometry_of_surfaces#Shape_operator
#    or https://en.wikipedia.org/wiki/Mean_curvature
#    '''
#    Zy, Zx  = numpy.gradient(Z)
#    Zxy, Zxx = numpy.gradient(Zx)
#    Zyy, _ = numpy.gradient(Zy)
#
#    H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
#    H = -H/(2*(Zx**2 + Zy**2 + 1)**(1.5))
#
#    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2
#
#    return H,K
