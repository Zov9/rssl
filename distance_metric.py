# l2 cosine wasstersin 
import torch
import numpy as np
import ot
import torch.nn.functional as F

def l2dist(tensor1,tensor2):
    l2_distance = torch.dist(tensor1, tensor2, p=2)
    return l2_distance

def cosdist(tensor1,tensor2):
    '''
    if not isinstance(tensor1, torch.Tensor):
        print('len of tensor1',len(tensor1),'len of tensor1[0]',len(tensor1[0]))
        tensor1 = [tensor1[key] for key in sorted(tensor1.keys())]
        print('len of tensor1',len(tensor1),'len of tensor1[0]',len(tensor1[0]))
        #print("tensor1 values after sorting keys:", tensor1)
        #tensor1 = torch.tensor(tensor1)
        tensor1 = torch.stack(tensor1, dim=0)
      
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = [tensor2[key] for key in sorted(tensor2.keys())]
        print("tensor2 values after sorting keys:", tensor2)
        tensor2 = torch.tensor(tensor2)
    '''
    cosine_similarity = F.cosine_similarity(tensor1, tensor2, dim=0)

    # Calculate cosine distance (which is 1 - cosine similarity)
    cosine_distance = 1 - cosine_similarity
    return torch.tensor(cosine_distance)

def wasdist(tensor1,tensor2):
    dist1_np = tensor1.numpy()
    dist2_np = tensor2.numpy()

    # Compute Wasserstein distance
    wasserstein_distance = ot.emd2([], [], np.outer(dist1_np, dist2_np))
    return torch.tensor(wasserstein_distance)