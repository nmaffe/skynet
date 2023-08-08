import torch
import Deepfillv2.libs.misc as misc

def RMSE(y1,y2,mins,maxs):
    '''
    Compute the RMSE between two denormalized images
    '''
#    x1=misc.pt_to_image(y1)

#    x2=misc.pt_to_image(y2)
    x1=misc.pt_to_image_denorm(y1,mins,maxs) # ha senso cosi com'è implementato?
    x2=misc.pt_to_image_denorm(y2,mins,maxs)
    
#    x1=(maxs-mins)*x1+mins
#    x2=(maxs-mins)*x2+mins

    return torch.sqrt(torch.mean((x1-x2)**2))
