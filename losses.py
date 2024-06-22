import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    #这里还得再改，因为原本的是监督对比学习，只有x 和 y 两个变量，但这里我有四个变量，x, u, labelx, pseudo label u

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签,尺寸是[batch_size].
            mask: 用于对比学习的mask,尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label,那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时,mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 
        # 这里我或许不用减 因为他这里都是一个数据里面的，所有自己没必要，我的话是另外一个数据里的，所以有必要
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        positives_mask = mask * logits_mask
        # 这些矩阵都是 batchsize*batchsize 的， 每一行， 以mask为例，每一行为1的值代表了batch所有样本中和它一类的样本
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言,(i,i)位置表示样本本身的相似度,对Loss是没用的,所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        # 这里是分母 是除了该样本外的所有样本，这里分同类和非同类两项相加
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    
class SupConLoss5(nn.Module): # for labeled data all the same default


    def __init__(self, temperature=0.5,distance= 0.5, scale_by_temperature=True):
        super(SupConLoss5, self).__init__()
        self.temperature = temperature
        self.distance = distance
        self.scale_by_temperature = scale_by_temperature

    def forward(self, worstk,features, labels=None, mask=None):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
     
        
       
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
       
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        positives_mask = mask * logits_mask
      
        negatives_mask = 1. - mask
      
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
   
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
      
        # loss
        loss = -log_probs
        
        loss = loss.mean()
        return loss

class SupConLoss6(nn.Module): # for labeled data like supconloss2


    def __init__(self, temperature=0.5,distance= 0.5, scale_by_temperature=True):
        super(SupConLoss6, self).__init__()
        self.temperature = temperature
        self.distance = distance
        self.scale_by_temperature = scale_by_temperature

    def forward(self, worstk,features, labels=None, mask=None):
        
        device = torch.device('cuda')
                  
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
     
        
        labels = torch.argmax(labels, dim=1)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        mask3 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
       
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)     
        positives_mask = mask * logits_mask
      
        negatives_mask = 1. - mask
      
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        num_positives_per_row_selected = mask3 * num_positives_per_row.t() 

        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
   
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        #log_probs = torch.sum(
        #    log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        
        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row_selected > 0] / num_positives_per_row_selected[num_positives_per_row_selected > 0]
      
        # loss
        loss = -log_probs
        
        loss = loss.mean()
        return loss
    


class SupConLoss7(nn.Module): # for labeled data like supconloss4


    def __init__(self, temperature=0.5,distance= 0.5, scale_by_temperature=True):
        super(SupConLoss7, self).__init__()
        self.temperature = temperature
        self.distance = distance
        self.scale_by_temperature = scale_by_temperature

    def forward(self, worstk,features, labels=None, mask=None):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
     
        
       
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
       
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        positives_mask = mask * logits_mask
      
        negatives_mask = 1. - mask
      
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
   
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
      
        # loss
        loss = -log_probs
        
        loss = loss.mean()
        return loss

class SupConLoss1(nn.Module):  #基本框架是这样 还得再check
    #这里还得再改，因为原本的是监督对比学习，只有x 和 y 两个变量，但这里我有四个变量，x, u, labelx, pseudo label u

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss1, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features1, features2,labels1 = None ,labels2 = None, mask=None):# x, u, labelx, labelu
       
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels1, labels2.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features1, features2.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
          
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
      
        # loss
        loss = -log_probs
        #if self.scale_by_temperature:
        #    loss *= self.temperature
        loss = loss.mean()
        return loss
    

class CLoss2(nn.Module):  #只对w类样本加closs  后面可以有一个supconloss3对所有样本都有closs但对w类额外加一项

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(CLoss2, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, tmask,worstk,features1, features2,labels1  ,labels2 , mask=None):# x, u, labelx, labelu
        #print('features1[0][:5]',features1[0][:5])
        #print('features2[0][:5]',features2[0][:5])
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        #print('features1[0][:5]',features1[0][:5])
        #print('features2[0][:5]',features2[0][:5])
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        num_class = len(labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels2, labels1.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        #mask3  based on worstk
        mask3 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels2[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)
        #40print('mask3',mask3)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features2, features1.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
        
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]   
        # first sum of all p in P(x) 
        #two sum  one for L(u) another for all u in D_u  2024/01/04
        #num_positives_per_row_selected = mask3 * num_positives_per_row #这里没问题吗 不需要再加一个 t() ?
        
        #print('Number of elements greater than 0 for original selection:',(num_positives_per_row > 0).sum().item())
        #print('Number of elements greater than 0 for  my  selection:',(num_positives_per_row_selected > 0).sum().item())
        
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        
        if torch.any(torch.isnan(log_probs)):
            #print('log_probs',log_probs)
            #print('logits1',logits1)
            #print('denominator',denominator)
            #print('torch.log(denominator)',torch.log(denominator))
            #print('anchor_dot_contrast1',anchor_dot_contrast1)
            #print('logits_max1',logits_max1)

            your_tensor_cpu = log_probs.cpu().detach().numpy()

            # File path to save the NumPy array
            file_path = "/data/lipeng/ABC/txt/nan_error_log.txt"

            # Save the NumPy array to a text file using with open and file.write()
            with open(file_path, 'a') as file:
                file.write('log_probs\n')
                for row in your_tensor_cpu:
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')
                file.write('logits1\n')
                for row in logits1.cpu().detach().numpy():
                        row_str = ', '.join(map(str, row))
                        file.write(row_str + '\n')
                file.write('denominator\n')
                for row in denominator.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')
                file.write('torch.log(denominator)\n')
                vab = torch.log(denominator)
                for row in vab.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n') 
                file.write('anchor_dot_contrast1\n')
                for row in anchor_dot_contrast1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')    
                file.write('logits_max1\n')
                for row in logits_max1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')   
                file.write('features2\n')
                for row in features2.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')  
                file.write('features1\n')
                for row in features1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')  
                mpv =  torch.matmul(features2, features1.T)
                file.write('matmul of feature2 and feature1.t\n')
                for row in mpv.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')
            raise ValueError("Log_prob has nan!")
        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[(num_positives_per_row> 0)&(mask3>0)&(tmask>0)] / num_positives_per_row[(num_positives_per_row > 0)&(mask3>0)&(tmask>0)]
        # loss
        loss = -log_probs
        loss = loss.mean()
        return loss
    
class CSLoss2(nn.Module):  #只对w类样本加closs  后面可以有一个supconloss3对所有样本都有closs但对w类额外加一项  
    #CSLoss2 仿照原始SUpconloss 一个输入一个label  closs2 closs3 是我的 两个输入两个label

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(CSLoss2, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, tmask,features1, labels1,worstk ):# x, u, labelx, labelu
        
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        batch_size = features1.shape[0]
        #num_class = len(labels1[0])
        #if len(labels1[0])==100:
        #    labels1 = torch.argmax(labels1, dim=1)
        labels1 = labels1.contiguous().view(-1, 1)
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels1, labels1.T).float().to(device)
        mask3 = torch.zeros(batch_size)
        for i in range(batch_size):
            if labels1[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features1, features1.T),
            self.temperature)  # Lu * Lu or Lx * Lx

        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        exp_logits1 = torch.exp(logits1)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask?         
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]   
        
        denominator = torch.sum(
        exp_logits1 * negatives_mask1, axis=1, keepdims=True) + torch.sum(
            exp_logits1 * positives_mask1, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        
        if torch.any(torch.isnan(log_probs)):            
            your_tensor_cpu = log_probs.cpu().detach().numpy()
            # File path to save the NumPy array
            file_path = "/data/lipeng/ABC/txt/nan_error_log.txt"
            # Save the NumPy array to a text file using with open and file.write()
            with open(file_path, 'a') as file:
                file.write('log_probs\n')
                for row in your_tensor_cpu:
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')
                file.write('logits1\n')
                for row in logits1.cpu().detach().numpy():
                        row_str = ', '.join(map(str, row))
                        file.write(row_str + '\n')
                file.write('denominator\n')
                for row in denominator.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')
                file.write('torch.log(denominator)\n')
                vab = torch.log(denominator)
                for row in vab.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n') 
                file.write('anchor_dot_contrast1\n')
                for row in anchor_dot_contrast1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')    
                file.write('logits_max1\n')
                for row in logits_max1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')   
                file.write('features2\n')
                for row in features1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')  
                file.write('features1\n')
                for row in features1.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')  
                mpv =  torch.matmul(features1, features1.T)
                file.write('matmul of feature2 and feature1.t\n')
                for row in mpv.cpu().detach().numpy():
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')
            raise ValueError("Log_prob has nan!")
        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        #print('log_probs1',log_probs)
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[(num_positives_per_row> 0)&(mask3>0)&(tmask>0)] / num_positives_per_row[(num_positives_per_row > 0)&(mask3>0)&(tmask>0)]
        #print('log_probs2',log_probs)
        # loss
        loss = -log_probs
        loss = loss.mean()
        #print('CSloss2',loss)
        return loss


    
class CLoss3(nn.Module):  #只对w类样本加closs  后面可以有一个supconloss3对所有样本都有closs但对w类额外加一项

    def __init__(self, temperature=0.5,distance = 0.5, scale_by_temperature=True):
        super(CLoss3, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.distance = distance
    def forward(self, tmask,worstk,features1, features2,labels1 = None ,labels2 = None, mask=None):# x, u, labelx, labelu
       
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        num_class = len(labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels2, labels1.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        #mask3  based on worstk
        mask3 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels2[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)
        mask4 = 1.-mask3
        
        mask4 = mask4.to(device)
        #40print('mask3',mask3)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features2, features1.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        anchor_dot_contrast3 = torch.div(
            torch.matmul(features2, features2.T)+self.distance,   #0.5 是distance, 可以考虑成类间距离或者类和最远样本的距离
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits_max3, _ = torch.max(anchor_dot_contrast3, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        logits3_1 = anchor_dot_contrast3 - logits_max2.detach()
        logits3_2 = anchor_dot_contrast3 - logits_max3.detach()
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
        exp_logits3 = torch.exp(logits3_2)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
        
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]   
        # first sum of all p in P(x) 
        #two sum  one for L(u) another for all u in D_u  2024/01/04
        #num_positives_per_row_selected = mask3 * num_positives_per_row #这里没问题吗 不需要再加一个 t() ?
        
        #print('Number of elements greater than 0 for original selection:',(num_positives_per_row > 0).sum().item())
        #print('Number of elements greater than 0 for  my  selection:',(num_positives_per_row_selected > 0).sum().item())
        
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        denominator1 = torch.sum(
        exp_logits3 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits3 * positives_mask2, axis=1, keepdims=True)
        
        log_probs = logits1 - torch.log(denominator)
        log_probs1 = logits1 - torch.log(denominator1)

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        if torch.any(torch.isnan(log_probs1)):
            raise ValueError("Log_prob1 has nan!")
        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[(num_positives_per_row> 0)&(mask3>0)&(tmask>0)] / num_positives_per_row[(num_positives_per_row > 0)&(mask3>0)&(tmask>0)]
        log_probs1 = torch.sum(
            log_probs1*positives_mask1 , axis=1)[(num_positives_per_row> 0)&(mask4>0)&(tmask>0)] / num_positives_per_row[(num_positives_per_row > 0)&(mask4>0)&(tmask>0)]
        # loss
        loss = -log_probs
        loss1 = -log_probs1
        loss = loss.mean()
        loss1 = loss1.mean()

        
        dp1 = torch.sum((num_positives_per_row   > 0)&(mask3>0)&(tmask>0)).item()
        dp2= torch.sum((num_positives_per_row   > 0)&(mask4>0)&(tmask>0)).item()
        if (dp1+dp2>0):
            ret = (dp1/(dp1+dp2))*loss+(dp2/(dp1+dp2))*loss1
        else:
            ret = loss

        return ret



class SupConLoss2(nn.Module):  #只对w类样本加closs  后面可以有一个supconloss3对所有样本都有closs但对w类额外加一项

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, tmask,worstk,features1, features2,labels1 = None ,labels2 = None, mask=None):# x, u, labelx, labelu
       
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        num_class = len(labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels1, labels2.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        #mask3  based on worstk
        mask3 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels2[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)
        #40print('mask3',mask3)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features1, features2.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
          
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]    
        num_positives_per_row_selected = mask3 * num_positives_per_row.t() 
        
        print('Number of elements greater than 0 for original selection:',(num_positives_per_row > 0).sum().item())
        print('Number of elements greater than 0 for  my  selection:',(num_positives_per_row_selected > 0).sum().item())
        
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[num_positives_per_row_selected > 0] / num_positives_per_row_selected[num_positives_per_row_selected > 0]
        # loss
        loss = -log_probs
        #if self.scale_by_temperature:
        #    loss *= self.temperature
        loss = loss.mean()
        return loss


class SupConLoss3(nn.Module):  #supconloss3对所有样本都有closs但对w类额外加一项  实际上是w类一项 + 非w类一项

    def __init__(self, temperature=0.5,distance=0.5, scale_by_temperature=True):
        super(SupConLoss3, self).__init__()
        self.temperature = temperature
        self.distance = distance
        self.scale_by_temperature = scale_by_temperature

    def forward(self, tmask,worstk,features1, features2,labels1 = None ,labels2 = None, mask=None):# x, u, labelx, labelu
       
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        num_class = len(labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels1, labels2.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        #mask3  based on worstk
        mask3 = torch.zeros(batch_size)
        #mask4  1 for mask and 0 for mask3, designed for samples not in worstk classes
        mask4 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels2[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)  
             
        mask4 = 1.-mask3
        
        mask4 = mask4.to(device)
        #40print('mask3',mask3)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features1, features2.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        anchor_dot_contrast3 = torch.div(
            torch.matmul(features2, features2.T)+self.distance,   #0.5 是distance, 可以考虑成类间距离或者类和最远样本的距离
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits_max3, _ = torch.max(anchor_dot_contrast3, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        #logits3 = anchor_dot_contrast3 - logits_max3.detach()
        logits3 = anchor_dot_contrast3 - logits_max2.detach()  #上下两行到底用哪个 可以都试试
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
        exp_logits3 = torch.exp(logits3)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
          
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]    
        num_positives_per_row_selected = mask3 * num_positives_per_row.t() 
        num_positives_per_row_unselected = mask4 * num_positives_per_row.t() 

        #print('Number of elements greater than 0 for original selection:',(num_positives_per_row > 0).sum().item())
        #print('Number of elements greater than 0 for  my  selection:',(num_positives_per_row_selected > 0).sum().item())
        
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        
        denominator1 = torch.sum(
        exp_logits3 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits3 * positives_mask2, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        
        log_probs1 = logits1 - torch.log(denominator1)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        


        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[num_positives_per_row_selected > 0] / num_positives_per_row_selected[num_positives_per_row_selected > 0]
        
        log_probs1 = torch.sum(
            log_probs1*positives_mask1 , axis=1)[num_positives_per_row_unselected > 0] / num_positives_per_row_unselected[num_positives_per_row_unselected > 0]
        # loss
        loss = -log_probs
        loss1 = -log_probs1
        #if self.scale_by_temperature:
        #    loss *= self.temperature
        loss = loss.mean()
        loss1 = loss1.mean()
        return loss+loss1

class SupConLoss3_(nn.Module):  #supconloss3 是+ distance 这是 -distance

    def __init__(self, temperature=0.5,distance=0.5, scale_by_temperature=True):
        super(SupConLoss3_, self).__init__()
        self.temperature = temperature
        self.distance = distance
        self.scale_by_temperature = scale_by_temperature

    def forward(self, tmask,worstk,features1, features2,labels1 = None ,labels2 = None, mask=None):# x, u, labelx, labelu
       
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        num_class = len(labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels1, labels2.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        #mask3  based on worstk
        mask3 = torch.zeros(batch_size)
        #mask4  1 for mask and 0 for mask3, designed for samples not in worstk classes
        mask4 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels2[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)  
             
        mask4 = 1.-mask3
        
        mask4 = mask4.to(device)
        #40print('mask3',mask3)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features1, features2.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        anchor_dot_contrast3 = torch.div(
            torch.matmul(features2, features2.T)-self.distance,   #0.5 是distance, 可以考虑成类间距离或者类和最远样本的距离
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits_max3, _ = torch.max(anchor_dot_contrast3, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        #logits3 = anchor_dot_contrast3 - logits_max3.detach()
        logits3 = anchor_dot_contrast3 - logits_max2.detach()  #上下两行到底用哪个 可以都试试
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
        exp_logits3 = torch.exp(logits3)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
          
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]    
        num_positives_per_row_selected = mask3 * num_positives_per_row.t() 
        num_positives_per_row_unselected = mask4 * num_positives_per_row.t() 

        #print('Number of elements greater than 0 for original selection:',(num_positives_per_row > 0).sum().item())
        #print('Number of elements greater than 0 for  my  selection:',(num_positives_per_row_selected > 0).sum().item())
        
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        
        denominator1 = torch.sum(
        exp_logits3 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits3 * positives_mask2, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        
        log_probs1 = logits1 - torch.log(denominator1)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        


        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[num_positives_per_row_selected > 0] / num_positives_per_row_selected[num_positives_per_row_selected > 0]
        
        log_probs1 = torch.sum(
            log_probs1*positives_mask1 , axis=1)[num_positives_per_row_unselected > 0] / num_positives_per_row_unselected[num_positives_per_row_unselected > 0]
        # loss
        loss = -log_probs
        loss1 = -log_probs1
        #if self.scale_by_temperature:
        #    loss *= self.temperature
        loss = loss.mean()
        loss1 = loss1.mean()
        return loss+loss1


class SupConLoss4(nn.Module):  #supconloss4对所有样本都有closs但对w类额外加一项  

    def __init__(self, temperature=0.5,distance=0.5, scale_by_temperature=True):
        super(SupConLoss4, self).__init__()
        self.temperature = temperature
        self.distance = distance
        self.scale_by_temperature = scale_by_temperature

    def forward(self, tmask,worstk,features1, features2,labels1 = None ,labels2 = None, mask=None):# x, u, labelx, labelu
       
        device = (torch.device('cuda'))
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        batch_size = features1.shape[0]
        batch_size2 = features2.shape[0]
        # 关于labels参数
        #if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
        #    raise ValueError('Cannot define both `labels` and `mask`') 
        #print('labels1[0]',labels1[0])
        num_class = len(labels1[0])
        labels1 = torch.argmax(labels1, dim=1)
        
        #print('labels1[0]',labels1[0])
        labels1 = labels1.contiguous().view(-1, 1)
        #labels2 = torch.argmax(labels2, dim=1)
        labels2 = labels2.contiguous().view(-1, 1)
        #print('label1 shape0:',labels1.shape[0],'label2 shape0:',labels2.shape[0],'feature1 shape0:',features1.shape[0],'feature2 shape0:',features2.shape[0])
        if labels1.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels1, labels2.T).float().to(device)
        mask2 = torch.eq(labels2, labels2.T).float().to(device)
        #mask3  based on worstk
        mask3 = torch.zeros(batch_size)
        #mask4  1 for mask and 0 for mask3, designed for samples not in worstk classes
        mask4 = torch.zeros(batch_size)
        for i in range(batch_size):
            #print('labels2[i]',labels2[i],'worstk',worstk)
            if labels2[i] in worstk:
                mask3[i] = 1
        mask3 = mask3.to(device)  
             
        mask4 = 1.-mask3
        
        mask4 = mask4.to(device)
        #40print('mask3',mask3)
        # compute logits
        anchor_dot_contrast1 = torch.div(                
            torch.matmul(features1, features2.T),
            self.temperature)  # Lx * Lu
        anchor_dot_contrast2 = torch.div(
            torch.matmul(features2, features2.T),
            self.temperature)  # Lu * Lu
        anchor_dot_contrast3 = torch.div(
            torch.matmul(features2, features2.T)+self.distance,   #0.5 是distance, 可以考虑成类间距离或者类和最远样本的距离
            self.temperature)  # Lu * Lu
        # for numerical stability
        logits_max1, _ = torch.max(anchor_dot_contrast1, dim=1, keepdim=True)
        logits_max2, _ = torch.max(anchor_dot_contrast2, dim=1, keepdim=True)
        logits_max3, _ = torch.max(anchor_dot_contrast3, dim=1, keepdim=True)
        logits1 = anchor_dot_contrast1 - logits_max1.detach()
        logits2 = anchor_dot_contrast2 - logits_max2.detach()
        #logits3 = anchor_dot_contrast3 - logits_max3.detach()
        logits3 = anchor_dot_contrast3 - logits_max2.detach()  #上下两行到底用哪个 可以都试试
        exp_logits1 = torch.exp(logits1)
        exp_logits2 = torch.exp(logits2)
        exp_logits3 = torch.exp(logits3)
      
        # 构建mask 
        #logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        logits_mask1 = torch.ones_like(mask).to(device)                          #label and unlabel
        

        logits_mask2 = torch.ones_like(mask2).to(device) - torch.eye(batch_size2).to(device)  #unlabel and unlabel

        positives_mask1 = mask * logits_mask1
        negatives_mask1 = 1. - mask                 #? 这里是不是有问题？ 是 1-mask 还是 1-positive mask? 
        #??????????????????????????????????????????????????????????????????????????????????????????????
        positives_mask2 = mask2 * logits_mask2
        negatives_mask2 = 1. - mask2
          
        num_positives_per_row  = torch.sum(positives_mask1 , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]    
        num_positives_per_row_selected = mask3 * num_positives_per_row.t() 
        num_positives_per_row_unselected = mask4 * num_positives_per_row.t() 

        #print('Number of elements greater than 0 for original selection:',(num_positives_per_row > 0).sum().item())
        #print('Number of elements greater than 0 for  my  selection:',(num_positives_per_row_selected > 0).sum().item())
        
        denominator = torch.sum(
        exp_logits2 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits2 * positives_mask2, axis=1, keepdims=True)  
        
        denominator1 = torch.sum(
        exp_logits3 * negatives_mask2, axis=1, keepdims=True) + torch.sum(
            exp_logits3 * positives_mask2, axis=1, keepdims=True)  
        
        log_probs = logits1 - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        
        log_probs1 = logits1 - torch.log(denominator1)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        


        #log_probs = torch.sum(
        #    log_probs*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        log_probs = torch.sum(
            log_probs*positives_mask1 , axis=1)[num_positives_per_row_selected > 0] / num_positives_per_row_selected[num_positives_per_row_selected > 0]
        
        log_probs1 = torch.sum(
            log_probs1*positives_mask1 , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        # loss
        loss = -log_probs
        loss1 = -log_probs1
        #if self.scale_by_temperature:
        #    loss *= self.temperature
        loss = loss.mean()
        loss1 = loss1.mean()
        return loss+loss1
    
def OECCloss(max_probs,ar):
    ret = max_probs-ar
    return torch.sum(ret**2)