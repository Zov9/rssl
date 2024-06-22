import torch
import torch.nn as nn
import torch.nn.functional as F

import concurrent.futures
from distance_metric import cosdist as spec_dist
import numpy as np

def logcm(cm,name,path):
    with open(path, 'a') as file:        
        file.write(name+'\n')
        for row in cm:
            formatted_row = [format(value, '.2f') for value in row]
            formatted_string = ' '.join(formatted_row)
            file.write(formatted_string+'\n')


def clsloss(input_tensor, target_tensor, num_class,instance_ct, mask=None):
    input_tensor = input_tensor.cuda()
    target_tensor = target_tensor.cuda()
    classwise_loss = []
    num_classes = num_class
    for i in range(num_classes):
        #print('target_tensor[0].size()=',target_tensor[0].size())
        #print('target_tensor[0]=',target_tensor[0])
        if target_tensor[0].numel()==1:
            class_mask = (target_tensor == i)
        else:
            class_mask = (target_tensor[:, i] == 1)

        if class_mask.any():
            ct1 = torch.sum(class_mask).item()
            instance_ct[i]+= ct1
            
            if mask is not None:
                #print('i',i,'\n')
                #print('target tensor\n',target_tensor,'\n')
                
                if target_tensor[0].numel()==1:
                    #print('len of class  mask',len(class_mask),'len of mask',len(mask))
                    #print('class_mask\n',class_mask,'\n')
                    #print('mask\n',mask,'\n')
                    #print('try',mask[class_mask])
                    #print('attempt success')
                    class_loss = torch.sum(F.cross_entropy(input_tensor[class_mask], target_tensor[class_mask], reduction='none') * mask[class_mask])
                    classwise_loss.append(class_loss.item())
                else:
                    #print('len of class  mask',len(class_mask),'len of mask',len(mask))
                    #print('class_mask\n',class_mask,'\n')
                    #print('mask\n',mask,'\n')
                    #print('try',mask[class_mask])
                    #print('attempt success')
                    class_loss = torch.sum(F.cross_entropy(input_tensor[class_mask], torch.argmax(target_tensor[class_mask], dim=1), reduction='none') * mask[class_mask])
                    classwise_loss.append(class_loss.item())
            else:
                if target_tensor[0].numel()==1:
                    class_loss = F.cross_entropy(input_tensor[class_mask], target_tensor[class_mask], reduction='sum')
                    classwise_loss.append(class_loss.item())
                else:
                    class_loss = F.cross_entropy(input_tensor[class_mask], torch.argmax(target_tensor[class_mask], dim=1), reduction='sum')
                    classwise_loss.append(class_loss.item())
        else:
            classwise_loss.append(0.0)

    return torch.tensor(classwise_loss)


def cal_fn_pair(cls_rep_x,cls_center_x,num_classes,name,txtpath):
    #here we store furthest distance and nearest distance
        fdist = torch.full((num_classes,num_classes),-1)
        ndist = torch.full((num_classes,num_classes),-1)
        #here we store furthest sample and nearest sample
        ftensor_x = torch.zeros(num_classes,num_classes,128)
        ntensor_x = torch.zeros(num_classes,num_classes,128)   

        furest_smp_dst = torch.full((num_classes,), -1)

        for i,  representations_i in cls_rep_x.items():
            for idx,rep in enumerate(representations_i):
                #update furthest distance between a sample and its class
                spec_dist_i = spec_dist(rep, cls_center_x)
                furest_smp_dst = max(furest_smp_dst, spec_dist_i)
                
                #loop through every class to find most distant and nearest pair  
                for j ,  representations_j in cls_rep_x.items():
                    #label_j, representations_j = list(class_representations.items())[j]
                    
                    if i==j:
                        continue
                    class_j_center = cls_center_x[j]
                    dist_ij = spec_dist(cls_rep_x[idx],class_j_center)
                    if fdist[j] == -1 or fdist[j] < dist_ij:
                        fdist[j] = dist_ij
                        ftensor_x[j] = rep
                    if ndist[j] == -1 or ndist[j] > dist_ij:
                        ndist[j] = dist_ij
                        ntensor_x[j] = rep
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i, rep_i in enumerate(ftensor_x):
            for j, rep_j in enumerate(rep_i):
                if i == j:
                    continue
                dist_f[j] = spec_dist(rep_j, ftensor_x[j])
                dist_n[j] = spec_dist(ntensor_x[j], ntensor_x[j])
        #maybe i can modify here to specify the situation about overlapping

        with open(txtpath, 'a') as file:
            
            file.write(name+"\n")
            file.write('Most farthest distance between i and j'+'\n')
            for row in dist_f:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Most nearest distance between i and j' + '\n')
            for row in dist_n:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Distance between most distant sample and class center' + '\n')
            #file.write("{}\n".format(furest_smp_dst))
            fsd = furest_smp_dst.cpu().detach().numpy()
            file.write(" ".join(map(str, fsd)) + "\n")

def cal_fn_pair1(cls_rep_x,cls_center_x,num_classes,name,txtpath):
    #here we store furthest distance and nearest distance
        fdist = torch.full((num_classes,num_classes),-1)
        ndist = torch.full((num_classes,num_classes),-1)
        #here we store furthest sample and nearest sample
        ftensor_x = torch.zeros(num_classes,num_classes,128)
        ntensor_x = torch.zeros(num_classes,num_classes,128)   

        furest_smp_dst = torch.full((num_classes,), -1)
        '''
        for i,  representations_i in cls_rep_x.items():#100
            for idx,rep in enumerate(representations_i):#num of sample in this cls
                #update furthest distance between a sample and its class
                spec_dist_i = spec_dist(rep, cls_center_x)
                furest_smp_dst = max(furest_smp_dst, spec_dist_i)
                
                #loop through every class to find most distant and nearest pair  
                for j ,  representations_j in cls_rep_x.items(): #100
                    #label_j, representations_j = list(class_representations.items())[j]
                    
                    if i==j:
                        continue
                    class_j_center = cls_center_x[j]
                    dist_ij = spec_dist(cls_rep_x[idx],class_j_center)
                    if fdist[j] == -1 or fdist[j] < dist_ij:
                        fdist[j] = dist_ij
                        ftensor_x[j] = rep
                    if ndist[j] == -1 or ndist[j] > dist_ij:
                        ndist[j] = dist_ij
                        ntensor_x[j] = rep
        '''
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(len(cls_rep_x)):
                #d_cls_rep_x = cls_rep_x.detach()
                #d_cls_center_x = cls_center_x.detach()
                f1 = executor.submit(tmp,cls_rep_x,cls_center_x,fdist,ndist,i,num_classes)
                ftensor_x,ntensor_x,fdist,ndist ,furest_smp_dst= f1.result()               
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i, rep_i in enumerate(ftensor_x):
            for j, rep_j in enumerate(rep_i):
                if i == j:
                    continue
                dist_f[j] = spec_dist(rep_j, ftensor_x[j])
                dist_n[j] = spec_dist(ntensor_x[j], ntensor_x[j])
        #maybe i can modify here to specify the situation about overlapping

        with open(txtpath, 'a') as file:
            
            file.write(name+"\n")
            file.write('Most farthest distance between i and j'+'\n')
            for row in dist_f:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Most nearest distance between i and j' + '\n')
            for row in dist_n:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Distance between most distant sample and class center' + '\n')
            #file.write("{}\n".format(furest_smp_dst))
            fsd = furest_smp_dst.cpu().detach().numpy()
            file.write(" ".join(map(str, fsd)) + "\n")

import torch.multiprocessing as mp

def cal_fn_pair2(cls_rep_x,cls_center_x,num_classes,name,txtpath):
    #here we store furthest distance and nearest distance
        fdist = torch.full((num_classes,num_classes),-1)
        ndist = torch.full((num_classes,num_classes),-1)
        #here we store furthest sample and nearest sample
        ftensor_x = torch.zeros(num_classes,num_classes,128)
        ntensor_x = torch.zeros(num_classes,num_classes,128)   

        furest_smp_dst = torch.full((num_classes,), -1)
        '''
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(len(cls_rep_x)):
                
                f1 = executor.submit(tmp,cls_rep_x,cls_center_x,fdist,ndist,i,num_classes)
                ftensor_x,ntensor_x,fdist,ndist ,furest_smp_dst= f1.result()               
        '''
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(10)
        
        for j in range(int(num_classes/10)):
            pool_list = []
            for i in range(10*j,10*j+10):
                res = pool.apply_async(tmp,args =  (cls_rep_x,cls_center_x,fdist,ndist,i,num_classes))
                pool_list.append(res)
            pool.close()
            pool.join()
            for res1 in pool_list:
                ftensor_x,ntensor_x,fdist,ndist ,furest_smp_dst= res1.get()               
            
                        
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i, rep_i in enumerate(ftensor_x):
            for j, rep_j in enumerate(rep_i):
                if i == j:
                    continue
                dist_f[j] = spec_dist(rep_j, ftensor_x[j])
                dist_n[j] = spec_dist(ntensor_x[j], ntensor_x[j])
        

        with open(txtpath, 'a') as file:
            
            file.write(name+"\n")
            file.write('Most farthest distance between i and j'+'\n')
            for row in dist_f:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Most nearest distance between i and j' + '\n')
            for row in dist_n:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Distance between most distant sample and class center' + '\n')
            
            fsd = furest_smp_dst.cpu().detach().numpy()
            file.write(" ".join(map(str, fsd)) + "\n")

def cal_fn_pair3(cls_rep_x,cls_center_x,num_classes,name,txtpath):
    #here we store furthest distance and nearest distance
        ttensor  = torch.ones(num_classes,num_classes)
        fdist = ttensor*-2
        ndist = ttensor*-2
        #here we store furthest sample and nearest sample
          
        ttensor1 = torch.ones(num_classes)
        furest_smp_dst = ttensor1*-2
        for i in range(num_classes):
            if not cls_rep_x[i]:
                continue
            furest_smp_dst[i],_ = cos_sim(cls_rep_x[i],cls_center_x[i],dim = 0)
            #print('furest_smp_dst[i]',furest_smp_dst[i])

        for i in range(num_classes):
            for j in range(i+1,num_classes):
                if j== i:
                    continue
                if not cls_rep_x[i]:
                    continue
                fdist[i][j],ndist[i][j] = cos_sim(cls_rep_x[i],cls_rep_x[j])
                #print("fdist[i][j],ndist[i][j]",fdist[i][j],ndist[i][j])
        #maybe i can modify here to specify the situation about overlapping

        with open(txtpath, 'a') as file:
            
            file.write(name+"\n")
            file.write('Most farthest distance between i and j'+'\n')
            for row in fdist:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Most nearest distance between i and j' + '\n')
            for row in ndist:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Distance between most distant sample and class center' + '\n')
            #file.write("{}\n".format(furest_smp_dst))
            fsd = furest_smp_dst.cpu().detach().numpy()
            file.write(" ".join(map(str, fsd)) + "\n")
        return furest_smp_dst    


def cal_fn_pair4(cls_rep_x,cls_center_x,num_classes,name,txtpath,mode_,key_):
    #here we store furthest distance and nearest distance
        ttensor  = torch.ones(num_classes,num_classes)
        fdist = ttensor*-2
        ndist = ttensor*-2
        #here we store furthest sample and nearest sample
          
        ttensor1 = torch.ones(num_classes)
        furest_smp_dst = ttensor1*-2
        for i in range(num_classes):
            if not cls_rep_x[i]:
                continue
            furest_smp_dst[i],_ = cos_sim1(cls_rep_x[i],cls_center_x[i],dim = 0,mode = mode_, k = key_)
            #print('furest_smp_dst[i]',furest_smp_dst[i])

        for i in range(num_classes):
            for j in range(i+1,num_classes):
                if j== i:
                    continue
                if not cls_rep_x[i]:
                    continue
                fdist[i][j],ndist[i][j] = cos_sim(cls_rep_x[i],cls_rep_x[j])
                #print("fdist[i][j],ndist[i][j]",fdist[i][j],ndist[i][j])
        #maybe i can modify here to specify the situation about overlapping
        '''
        with open(txtpath, 'a') as file:
            
            file.write(name+"\n")
            file.write('Most farthest distance between i and j'+'\n')
            for row in fdist:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Most nearest distance between i and j' + '\n')
            for row in ndist:
                file.write(" ".join(map(str, row.cpu().detach().numpy())) + "\n")
            file.write('Distance between most distant sample and class center' + '\n')
            #file.write("{}\n".format(furest_smp_dst))
            fsd = furest_smp_dst.cpu().detach().numpy()
            file.write(" ".join(map(str, fsd)) + "\n")
        '''
        return furest_smp_dst    


def cos_sim(A,B,dim = 1):
    if isinstance(A,list):
        A = torch.stack(A,dim = 0)
    if isinstance(B,list):
        try:
            B = torch.stack(B ,dim = 0)
        except:
            return -2,-2
    if dim == 0:
        norms_A = torch.norm(A)
        norms_B = torch.norm(B)
        dot_product = torch.matmul(A, B.t())
        norms_product = norms_A* norms_B
    else:
        norms_A = torch.norm(A, dim=1, keepdim=True)
        norms_B = torch.norm(B, dim=1, keepdim=True)

    # Compute the dot product of the matrices
        dot_product = torch.mm(A, B.t())

    # Compute the product of the norms
        norms_product = torch.mm(norms_A, norms_B.t())

    # Compute the cosine similarity
    cosine_similarity = torch.div(dot_product , norms_product)
    #print('cosine_similarity[:10]',cosine_similarity[:5],'\ndot product',dot_product[:5],'\n norms product',norms_product[:5])
    #max similarity equals to min distance, same for the opposite
    mind = torch.max(cosine_similarity)
    maxd = torch.min(cosine_similarity)
    #print('mind and maxd',mind,maxd)
    #print('cosine_similarity[:1]',cosine_similarity[0])
    return maxd,mind

def cos_sim1(A, B, dim=1, mode=None, k=None):
    # Check if k and mode are provided
    if k is None or mode is None:
        raise ValueError("Both 'k' and 'mode' must be specified.")

    
    # Existing code to compute cosine_similarity...
    if isinstance(A,list):
        A = torch.stack(A,dim = 0)
    if isinstance(B,list):
        try:
            B = torch.stack(B ,dim = 0)
        except:
            return -2,-2
    if dim == 0:
        norms_A = torch.norm(A)
        norms_B = torch.norm(B)
        dot_product = torch.matmul(A, B.t())
        norms_product = norms_A* norms_B
    else:
        norms_A = torch.norm(A, dim=1, keepdim=True)
        norms_B = torch.norm(B, dim=1, keepdim=True)

    # Compute the dot product of the matrices
        dot_product = torch.mm(A, B.t())

    # Compute the product of the norms
        norms_product = torch.mm(norms_A, norms_B.t())

    # Compute the cosine similarity
    cosine_similarity = torch.div(dot_product , norms_product)
    mind = torch.max(cosine_similarity)
    # Handle the new 'mode' functionality
    '''
    #mind = torch.max(cosine_similarity)
    #maxd = torch.min(cosine_similarity)
    if mode == 'avg':
        # Sort the cosine similarities in descending order (furthest first)
        sorted_similarities, _ = torch.sort(cosine_similarity)  #simlarity from small to big
        # Compute the average of the top k furthest distances
        maxd = torch.mean(sorted_similarities[:k])
    elif mode == 'furst':
        # Sort the cosine similarities in descending order (furthest first)
        sorted_similarities, _ = torch.sort(cosine_similarity)
        # Get the kth furthest distance
        maxd = sorted_similarities[k-1]
    else:
        raise ValueError("Invalid mode. Supported modes: 'avg' or 'furst'.")
    '''

    if len(cosine_similarity) < k:
        print(f"Warning: 'k' ({k}) is greater than the length of cosine_similarity ({len(cosine_similarity)}). Adjusting 'k' to {len(cosine_similarity)}.")
        k = len(cosine_similarity)

    # Proceed with the rest of the code...
    if mode == 'avg':
        # Sort the cosine similarities in descending order (furthest first)
        sorted_similarities, _ = torch.sort(cosine_similarity, descending=True)
        # Compute the average of the top k furthest distances
        maxd = torch.mean(sorted_similarities[:k]) if k > 0 else torch.tensor(float('nan'))
    elif mode == 'furst':
        # Sort the cosine similarities in descending order (furthest first)
        sorted_similarities, _ = torch.sort(cosine_similarity, descending=True)
        # Get the kth furthest distance or the last element if k is out of bounds
        maxd = sorted_similarities[k-1] if k > 0 else sorted_similarities[-1]
    else:
        raise ValueError("Invalid mode. Supported modes: 'avg' or 'furst'.")

    # Rest of the existing code...
    return maxd, mind
                
def tmp(cls_rep_x,cls_center_x,ret3_o,ret4_o,i,num_class):
            ret1 = torch.zeros(num_class,128)#furthest
            ret2 = torch.zeros(num_class,128)#nearest
            ret3 = ret3_o#fdist
            ret4 = ret4_o#ndist
            max_dist = -1
            representations_i = cls_rep_x#100
            for idx,rep in enumerate(representations_i):#num of sample in this cls
                #update furthest distance between a sample and its class
                #rep = rep.detach()
                #class_i_center = cls_center_x.detach()
                class_i_center = cls_center_x
                spec_dist_i = spec_dist(rep, class_i_center)
                max_dist = max(max_dist, spec_dist_i)
                
                #loop through every class to find most distant and nearest pair  
                for j  in range(num_class): #100
                    #label_j, representations_j = list(class_representations.items())[j]
                    
                    if i==j:
                        continue
                    class_j_center = cls_center_x[j]
                    dist_ij = spec_dist(rep,class_j_center)
                    if ret3[j] == -1 or ret3[j] < dist_ij:
                        ret3[j] = dist_ij
                        ret1[j] = rep
                    if ret4[j] == -1 or ret4[j] > dist_ij:
                        ret4[j] = dist_ij
                        ret2[j] = rep
            return ret1,ret2,ret3,ret4,max_dist

def cal_cent_dist(cls_center_x,spec_dist,name,txtpath):
    dist = torch.zeros(len(cls_center_x),len(cls_center_x))
    for i in range(len(cls_center_x)):
        for j in range(i,len(cls_center_x)):
            dist[i][j] = spec_dist(cls_center_x[i],cls_center_x[j])
    '''        
    with open(txtpath, 'a') as file:           
            file.write(name +"\n")
            file.write('Distance between class center i and class center j'+'\n')
            for row in dist:
                row = row.cpu().detach().numpy()
                file.write(" ".join(map(str, row)) + "\n")
    '''
    return dist

def calculate_confusion_matrix(y_true, y_pred,args, threshold=0.5):
    if args.dataset=='cifar100':
        num_classes = 100
    else:
        num_classes = 10
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    
    for true_tensor, pred_tensor in zip(y_true, y_pred):
        for true_label, probabilities in zip(true_tensor, pred_tensor):
            similarity = [prob for prob in probabilities]
            for i, sim_score in enumerate(similarity):
                try:
                    confusion_matrix[true_label] += sim_score
                except:
                    sim_score = sim_score.detach().cpu().numpy()   
    return confusion_matrix

import math
# for finding worst classes

def overlap_area(c1_radius, c2_radius, distance_between_centers):
    # No overlap
    if distance_between_centers >= c1_radius + c2_radius:
        return 0.0

    # One circle completely inside the other
    if c1_radius < distance_between_centers - c2_radius:
        return math.pi * c1_radius**2

    if c2_radius < distance_between_centers - c1_radius:
        return math.pi * c2_radius**2

    # Partial overlap
    d = distance_between_centers
    r1 = c1_radius
    r2 = c2_radius

    # Check if circles are completely separate
    if d > r1 + r2:
        return 0.0

    a1 = r1**2 * math.acos(max(-1, min((d**2 + r1**2 - r2**2) / (2 * d * r1), 1)))
    a2 = r2**2 * math.acos(max(-1, min((d**2 + r2**2 - r1**2) / (2 * d * r2), 1)))
    a3 = 0.5 * math.sqrt(max(0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))

    return a1 + a2 - a3

def row_sum(a):
    result = [] # An empty list to store the sums
    for i in range(len(a)): # Loop over the rows
        row_sum = 0 # A variable to store the sum for the current row
        cnt = 0
        for j in range(len(a[i])): # Loop over the columns
            if i != j and a[i][j] != -1: # Check if the element is not on the diagonal and not equal to -1
                row_sum += a[i][j] # Add the element to the row sum
                cnt+=1
        try:
            result.append(row_sum/cnt) # Append the row sum to the result list
        except:
            result.append(-1)
    return result # Return the result list

def worstk(k,usdist,Lu_part):  #usdist -- distance between classes for unlabled selected data   //   Lu_part  -- furthest distance between sample and class center for u s data
    overlap_percent_upart = usdist
    for j in range(100):
        for k in range(100):
            if j<k:
                
                try:
                    dist2 = overlap_area(1-Lu_part[j],1-Lu_part[k],usdist[j][k])
                    overlap_percent_upart[j][k] = dist2/(math.pi*(1-Lu_part[j])*(1-Lu_part[j]))
                    overlap_percent_upart[k][j] = dist2/(math.pi*(1-Lu_part[k])*(1-Lu_part[k]))
                except:
                    overlap_percent_upart[j][k] = -1
                    overlap_percent_upart[k][j] = -1

    overlap_avg_upart = Lu_part
    overlap_avg_upart = row_sum(overlap_percent_upart)
    asc_indices2 = np.argsort(overlap_avg_upart)
    desc_indices2 = asc_indices2[::-1]
    ret = desc_indices2[:20]
    info_pairs = [(i, overlap_avg_upart[i]) for i in desc_indices2]
    print('info_pairs',info_pairs)
    print('returned from worstk func,',ret)
    return tuple(ret),info_pairs



def compute_py(train_loader, args):
    """compute the base probabilities"""
    device = (torch.device('cuda'))
    label_freq = {}
    for i, (inputs, labell,_) in enumerate(train_loader):
        labell = labell.to(device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(device)
    #print('Label-freq-array',label_freq_array)
    return label_freq_array

def compute_adjustment_by_py(py, tro, args):
    device = (torch.device('cuda'))
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(device)
    return adjustments


def compute_py1(train_loader, wk):
    """compute the base probabilities"""
    device = (torch.device('cuda'))
    label_freq = {}
    for i, (inputs, labell,_) in enumerate(train_loader):
        labell = labell.to(device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))

    original_sum = label_freq_array.sum()
    adjusted_value = 0.01 * original_sum
    for class_label in wk:
        if class_label in label_freq:
            tmp = 0.3*label_freq_array[class_label]
            label_freq_array[class_label] -= tmp


    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(device)
    #print('Label-freq-array',label_freq_array)
    return label_freq_array

def compute_py2(train_loader, wk):
    """compute the base probabilities"""
    device = (torch.device('cuda'))
    label_freq = {}
    for i, (inputs, labell,_) in enumerate(train_loader):
        labell = labell.to(device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))

    original_sum = label_freq_array.sum()
    adjusted_value = 0.02 * original_sum
    for class_label in wk:
        if class_label in label_freq:
            label_freq_array[class_label] -= adjusted_value

    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(device)
    #print('Label-freq-array',label_freq_array)
    return label_freq_array

from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)