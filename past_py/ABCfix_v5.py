# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import numpy as np
import wideresnetwithABC as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy import optimize
from sklearn.metrics import confusion_matrix
from collections import Counter

from distance_metric import cosdist as spec_dist
from datetime import datetime

import concurrent.futures

#txtpath = "/data/lipeng/ABC/txt/try100_1103_1453_dist.txt"
txtpath = '/data/lipeng/ABC/txt/cf100_1212_1423_1.txt'


parser = argparse.ArgumentParser(description='PyTorch fixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--num_max', type=int, default=1500,
                        help='Number of samples in the maximal class')
parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
parser.add_argument('--step', action='store_true', help='Type of class-imbalance')
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')
parser.add_argument('--num_val', type=int, default=10,
                        help='Number of validation data')
# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
#dataset and imbalanced type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
if args.dataset=='cifar10':
    import dataset.fix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10
elif args.dataset=='svhn':
    import dataset.fix_svhn as dataset
    print(f'==> Preparing imbalanced SVHN')
    num_class = 10
elif args.dataset=='cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
# np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    #U_SAMPLES_PER_CLASS = make_imb_data((100-args.label_ratio)/args.label_ratio * args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.label_ratio * args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    ir2=N_SAMPLES_PER_CLASS[-1]/np.array(N_SAMPLES_PER_CLASS)
    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set,test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    elif args.dataset == 'svhn':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_SVHN('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
    elif args.dataset =='cifar100':

        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2 with abc")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=num_class)
        model = model.cuda()

        params = list(model.parameters())
        if ema:
            for param in params:
                param.detach_()

        return model, params

    model, params = create_model()
    ema_model,  _ = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'ABCfix-' + args.dataset
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'abcloss','Test Loss', 'Test Acc.'])

    #==================
    #unlabeled_trainloader
    N = len(unlabeled_trainloader.dataset.indices)
    
    learning_status = [-1] * N
    #==================
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))


        # Training part
        train_loss, train_loss_x, train_loss_u, abcloss = train(labeled_trainloader,
                                                                                                unlabeled_trainloader,
                                                                                                model, optimizer,
                                                                                                ema_optimizer,
                                                                                                train_criterion,
                                                                                                epoch,ir2,
                                                                                                learning_status=learning_status,N = N)

        test_loss, test_acc, testclassacc = validate(test_loader, ema_model, criterion, mode='Test Stats ',epoch = epoch)
        if args.dataset == 'cifar10':
            print("each class accuracy test", testclassacc, testclassacc.mean(),testclassacc[:5].mean(),testclassacc[5:].mean())
        elif args.dataset == 'svhn':
            print("each class accuracy test", testclassacc, testclassacc.mean(), testclassacc[:5].mean(),testclassacc[5:].mean())
        elif args.dataset == 'cifar100':
            print("each class accuracy test", testclassacc, testclassacc.mean(), testclassacc[:50].mean(),testclassacc[50:].mean())

        logger.append([train_loss, train_loss_x, train_loss_u,abcloss, test_loss, test_acc])

        # Save models
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),

                'optimizer' : optimizer.state_dict(),
            }, epoch + 1)

    logger.close()


def calculate_confusion_matrix(y_true, y_pred, threshold=0.5):
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
                    confusion_matrix[true_label][i] += sim_score
                except:
                    sim_score = sim_score.detach().cpu().numpy()   
    return confusion_matrix



def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,ir2,learning_status,N):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_abc = AverageMeter()
    end = time.time()
    #========================
    #num_class = 100
    #===========================

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    u_list = []
    u_target = []
    u_pseu = []
    u_pred = []
    l_target = []
    l_pred = []


    u_cm_pred_w = []
    u_cm_pred_s = []
    u_cm_target = []
    u_cm_pseu = []
    u_cm_pseu1 = []
    l_cm_target = []
    l_cm_pred = []
    u_cm_target95 = []
    u_cm_pred_s95 = []

    cls_center_x = {}
    cls_center_ureal = {}
    cls_center_upsu_part = {}
    cls_center_upsu_all = {}
    cls_rep_x = {i: [] for i in range(num_class)}
    cls_rep_ureal = {i: [] for i in range(num_class)}
    cls_rep_upsu_part =  {i: [] for i in range(num_class)}
    cls_rep_upsu_all =  {i: [] for i in range(num_class)}


    Lx_w = torch.zeros(num_class)
    Lu_w = torch.zeros(num_class)
    Labc_w = torch.zeros(num_class)
    Lu_real_w1= torch.zeros(num_class)
    #Lu_real_w2= torch.zeros(num_class)
    
    num_alli = 0
    num_allu = 0

    lxct = [0]*num_class
    luct = [0]*num_class
    lurct = [0]*num_class
    for batch_idx in range(args.val_iteration):
        try:
            #inputs_x, targets_x, _ = labeled_train_iter.next()
            inputs_x, targets_x, _ = next(labeled_train_iter)
            l_target.append(targets_x)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            #inputs_x, targets_x, _ = labeled_train_iter.next()
            inputs_x, targets_x, _ = next(labeled_train_iter)
            l_target.append(targets_x)
        try:
            (inputs_u, inputs_u2, inputs_u3), targets_su, idx_u = next(unlabeled_train_iter)
            u_list.append(inputs_u)
            u_target.append(targets_su)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), targets_su, idx_u = next(unlabeled_train_iter)
            u_list.append(inputs_u)
            u_target.append(targets_su)
        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)

        l_cm_target.append(targets_x)#!!!
        u_cm_target.append(targets_su)#!!!

        targets_su2 = torch.zeros(batch_size, num_class).scatter_(1, targets_su.view(-1,1), 1)
        targets_su2 = targets_su2.cuda()

        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3 ,targets_su = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(),targets_su.cuda()

        #=====
        counter = Counter(learning_status)


        num_unused = counter[-1]
        if num_unused != N:
            max_counter = max([counter[c] for c in range(num_class)])
            if max_counter < num_unused:
                # normalize with eq.11
                sum_counter = sum([counter[c] for c in range(num_class)])                    
                denominator = max(max_counter, N - sum_counter)
            else:
                denominator = max_counter
            # threshold per class
            #for c in range(num_class):
            beta = [counter[c] / denominator for c in range(num_class)]


            result_arr = [10.0, 9.733333333333333, 9.533333333333333, 9.266666666666667, 9.066666666666666, 8.866666666666667, 8.666666666666666, 8.466666666666667, 8.266666666666667, 8.066666666666666, 7.866666666666666, 7.733333333333333, 7.533333333333333, 7.333333333333333, 7.2, 7.0, 6.866666666666666, 6.733333333333333, 6.533333333333333, 6.4, 6.266666666666667, 6.133333333333334, 5.933333333333334, 5.8, 5.666666666666667, 5.533333333333333, 5.4, 5.333333333333333, 5.2, 5.066666666666666, 4.933333333333334, 4.8, 4.733333333333333, 4.6, 4.533333333333333, 4.4, 4.266666666666667, 4.2, 4.066666666666666, 4.0, 3.933333333333333, 3.8, 3.7333333333333334, 3.6666666666666665, 3.533333333333333, 3.466666666666667, 3.4, 3.3333333333333335, 3.2666666666666666, 3.1333333333333333, 3.066666666666667, 3.0, 2.933333333333333, 2.8666666666666667, 2.8, 2.7333333333333334, 2.6666666666666665, 2.6, 2.533333333333333, 2.533333333333333, 2.466666666666667, 2.4, 2.3333333333333335, 2.2666666666666666, 2.2, 2.2, 2.1333333333333333, 2.066666666666667, 2.0, 2.0, 1.9333333333333333, 1.8666666666666667, 1.8666666666666667, 1.8, 1.7333333333333334, 1.7333333333333334, 1.6666666666666667, 1.6666666666666667, 1.6, 1.5333333333333334, 1.5333333333333334, 1.4666666666666666, 1.4666666666666666, 1.4, 1.4, 1.3333333333333333, 1.3333333333333333, 1.2666666666666666, 1.2666666666666666, 1.2, 1.2, 1.2, 1.1333333333333333, 1.1333333333333333, 1.0666666666666667, 1.0666666666666667, 1.0666666666666667, 1.0, 1.0, 1.0]

            
            # Creating a new counter without key -1
            counter1 = Counter({k: v / result_arr[k] for k, v in counter.items() if k != -1})


            smallest_5 = counter1.most_common()[:-6:-1]  # Reverse the Counter to get smallest
            smallest_10 = counter1.most_common()[:-11:-1]  
            smallest_20 = counter1.most_common()[:-21:-1]  
            # Open the file in 'a' (append) mode and write the information
            
        #=====

        # Generate the pseudo labels
        #
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            q1=model(inputs_u)
            outputs_u= model.classify(q1)
            targets_u2 = torch.softmax(outputs_u, dim=1).detach()

            
            #==========================================================
           
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！    

        targets_u = torch.argmax(targets_u2, dim=1)

        u_pred.append(targets_u2)

        q = model(inputs_x)
        
        q2 = model(inputs_u2)
        q3 = model(inputs_u3)

        max_p, p_hat = torch.max(targets_u2, dim=1)
        #print("len(max_p)",len(max_p),"max_p[0]",max_p[0])
        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(0.95)
        org_mask = select_mask
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)
        all_rtargets = torch.cat([targets_x2,targets_su2,targets_su2],dim=0)

        logits_x=model.classify(q)
        logits_u1=model.classify(q2)
        logits_u2=model.classify(q3)
        logits_u = torch.cat([logits_u1,logits_u2],dim=0)

        maskforbalance = torch.bernoulli(torch.sum(targets_x2 * torch.tensor(ir2).cuda(0), dim=1).detach())

        logit = model.classify2(q)#labeled sample
        logitu1 = model.classify2(q1)#weak aug unlabel
        logitu2 = model.classify2(q2)
        logitu3 = model.classify2(q3)

        logits = F.softmax(logit)#labeled sample
        logitsu1 = F.softmax(logitu1)#weak aug unlabel

        with torch.no_grad():
            max_prob, hard_label = torch.max(logitu1, dim=1)
            over_threshold = max_prob >= 0.95
            if over_threshold.any():
                idx_u = idx_u.cuda()
                sample_index = idx_u[over_threshold].tolist()
                pseudo_label = hard_label[over_threshold].tolist()
                for i, l in zip(sample_index, pseudo_label):
                    learning_status[i] = l





        u_cm_pred_w.append(logitsu1)#!!!
        l_cm_pred.append(logits)#!!!


        max_p2, label_u = torch.max(logitsu1, dim=1)
        select_mask2 = max_p2.ge(0.95)
        label_u = torch.zeros(batch_size, num_class).scatter_(1, label_u.cpu().view(-1, 1), 1)
        ir22 = 1 - (epoch / 500) * (1 - ir2)
        maskforbalanceu = torch.bernoulli(torch.sum(label_u.cuda(0) * torch.tensor(ir22).cuda(0), dim=1).detach())
        logitsu2 = F.softmax(logitu2)

        u_cm_pred_s.append(logitsu2)#!!!

        logitsu3 = F.softmax(logitu3)


        #---------------------------------------------------------------
        # targets_x2   q   label
        # targets_su   q1  real
        # targets_u2   q1  pseu    
        # targets_u2   q1  pseu select_mask   

        num_alli+=len(q)
        num_allu+=len(q1)

        targets_x2_1 = torch.tensor([torch.argmax(tensor).item() for tensor in targets_x2])      
        for i in range(num_class):
                #print('targets_x2_1',targets_x2_1)
                class_mask = (targets_x2_1 == i)
                #print('class_mask',class_mask)
                if class_mask.any():                                                           
                    cls_rep_x[i].extend(q[class_mask].detach())
                    #print('cls_rep_x len',len(cls_rep_x),'len cls rep x 0',len(cls_rep_x[0]),'cls_rep_x[0][:10]',cls_rep_x[0][:10])
            
        for i in range(num_class):
                class_mask = (targets_u == i)
                if class_mask.any():                                                           
                    cls_rep_upsu_all[i].extend(q1[class_mask].detach())
                                                    
        for i in range(num_class):
                class_mask = (targets_u == i)      
                #print('len(class_mask)',len(class_mask),'len(select_mask)',len(select_mask),'class_mask[0]',class_mask[0],'select_mask[0]',select_mask[0])          
                mask_h1 = org_mask*class_mask                
                if mask_h1.any():                                                                           
                    cls_rep_upsu_part[i].extend(q1[mask_h1.bool()].detach())
               
        for i in range(num_class):
                class_mask = (targets_su == i)
                if class_mask.any():                                                           
                    cls_rep_ureal[i].extend(q1[class_mask].detach())

        #---------------------------------------------------------------






        abcloss = -torch.mean(maskforbalance * torch.sum(torch.log(logits) * targets_x2.cuda(0), dim=1))
        abcloss1 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu2) * logitsu1.cuda(0).detach(), dim=1))

        abcloss2 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu3) * logitsu1.cuda(0).detach(), dim=1))

        totalabcloss=abcloss+abcloss1+abcloss2
        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        #print("lx,lu,abcloss",Lx,Lu,totalabcloss)
        loss = Lx + Lu+totalabcloss

        



        lxw = clsloss(logit, targets_x,num_class,lxct)
        #print('logits_u  len',len(logits_u),'select mask  len',len(select_mask))
        luw1 = clsloss(logitu2, hard_label,num_class,luct,org_mask)
        luw2 = clsloss(logitu3, hard_label,num_class,luct,org_mask)
        Lx_w+=lxw
        Lu_w= Lu_w+luw1+luw2
        #print('len outputs u ',len(outputs_u),outputs_u[0])
        #print('len targets_su',len(targets_su),targets_su[0])
        lur = clsloss(logitu1,targets_su,num_class,lurct)
        Lu_real_w1+=lur
        
        

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_abc.update(abcloss.item(), inputs_x.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}| Loss_m: {loss_m:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
            loss_m=losses_abc.avg,
                    )
        bar.next()
    bar.finish()

    print('num of labeled tensors',num_alli)
    print('num of unlabeled tensors',num_allu)
    print('lxct',lxct)
    print('luct',luct)
    print('lurct',lurct)
    with open(txtpath,'a') as file:
            file.write('lxct\n')
            file.write(','.join(map(str, lxct))) 
            file.write('\nluct\n')
            file.write(','.join(map(str, luct))) 
            file.write('\nlurct\n')
            file.write(','.join(map(str, lurct)))  
            file.write('\n')
            
            

    Lx_w = Lx_w.tolist()    
    Lu_w = Lu_w.tolist()
    Lu_real_w1= Lu_real_w1.tolist()
    Labc_w = Labc_w.tolist()



    ############################################
        ############################################
        #cls_center_x,   cls_center_ureal,    cls_center_upsu_part,    cls_center_upsu_all = {}
        #cls_rep_x,      cls_rep_ureal,       cls_rep_upsu_part,       cls_rep_upsu_all = {}
    current_time = datetime.now()
    print("523 Current Time:", current_time)
    for label, representations in cls_rep_x.items():
            class_center = torch.mean(torch.stack(representations), dim=0)
            #print('class_center',len(class_center))
            #print(class_center)
            cls_center_x[label] = class_center
    for label, representations in cls_rep_ureal.items():
            class_center = torch.mean(torch.stack(representations), dim=0)
            cls_center_ureal[label] = class_center
    for label, representations in cls_rep_upsu_part.items():
            try:
                class_center = torch.mean(torch.stack(representations), dim=0)

            except:
                class_center = torch.zeros(128).cuda()
            cls_center_upsu_part[label] = class_center
    for label, representations in cls_rep_upsu_all.items():
            try:
                class_center = torch.mean(torch.stack(representations), dim=0)
            except:
                class_center = torch.zeros(128).cuda()
            cls_center_upsu_all[label] = class_center
    current_time = datetime.now()
    print("546 Current Time:", current_time)
    cal_cent_dist(cls_center_x,spec_dist,'labeled center')
    cal_cent_dist(cls_center_ureal,spec_dist,'unlabeled real center')
    cal_cent_dist(cls_center_upsu_part,spec_dist,'unlabeled selected center')
    cal_cent_dist(cls_center_upsu_all,spec_dist,'unlabeled all center')

        # Calculate the nearest and furthest samples for each pair of classes
    num_classes = num_class
    current_time = datetime.now()    

    print("555 Current Time:", current_time)
    
    cal_fn_pair3(cls_rep_x,cls_center_x,num_classes,'cls_rep_x')

    current_time = datetime.now()
    print("558 Current Time:", current_time)
    
    cal_fn_pair3(cls_rep_ureal,cls_center_ureal,num_classes,'cls_rep_ureal')    

    current_time = datetime.now()
    print("561 Current Time:", current_time)
  
    cal_fn_pair3(cls_rep_upsu_part,cls_center_upsu_part,num_classes,'cls_rep_upsu_part')
   
    current_time = datetime.now()
    print("564 Current Time:", current_time)
                           
    cal_fn_pair3(cls_rep_upsu_all,cls_center_upsu_all,num_classes,'cls_rep_upsu_all')
                
    current_time = datetime.now()
    print("567 Current Time:", current_time)

        ############################################
        ############################################
                
    with open(txtpath,'a') as file:
            file.write('Lx_w\n') 
            for item in Lx_w:
                file.write("%s," % item)
            file.write('\n') 

            file.write('Lu_w\n') 
            for item in Lu_w:
                file.write("%s," % item)
            file.write('\n') 

            file.write('Lu_real_w1\n') 
            for item in Lu_real_w1:
                file.write("%s," % item)
            file.write('\n') 

    with open(txtpath, 'a') as file:
                file.write("Epoch: "+str(epoch)+'\n')
                file.write("Counter and beta:\n")
                file.write(f"Counter: {dict(counter)}\n")
                file.write(f"Counter1: {dict(counter1)}\n")
                file.write(f"Beta: {beta}\n")
                file.write("5 Smallest Classes:\n")
                for class_idx, count in smallest_5:
                    file.write(f"Class {class_idx}: Counter Value = {count}\n")
                file.write("10 Smallest Classes:\n")
                for class_idx, count in smallest_10:
                    file.write(f"Class {class_idx}: Counter Value = {count}\n")
                file.write("20 Smallest Classes:\n")
                for class_idx, count in smallest_20:
                    file.write(f"Class {class_idx}: Counter Value = {count}\n")        

    return (losses.avg, losses_x.avg, losses_u.avg, losses_abc.avg)

def validate(valloader, model, criterion, mode, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Initialize lists to store class-wise precision and recall
    classwise_precision = []
    classwise_recall = []

    # switch to evaluate mode
    model.eval()

    accperclass = np.zeros((num_class))

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            q = model(inputs)
            outputs2 = model.classify2(q)
            loss = criterion(outputs2, targets)

            unbiasedscore = F.softmax(outputs2)
            unbiased = torch.argmax(unbiasedscore, dim=1)
            outputs2onehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, unbiased.cpu().view(-1, 1).long(), 1)

            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(np.int64)

            # Append targets and outputs for later calculation of precision and recall
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(unbiased.cpu().numpy())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs2, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()

    if args.dataset == 'cifar10':
        accperclass = accperclass / 1000
    elif args.dataset == 'svhn':
        accperclass = accperclass / 1500
    elif args.dataset == 'cifar100':
        accperclass = accperclass / 100

    # Calculate confusion matrix from all_targets and all_outputs
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    cm = confusion_matrix(all_targets, all_outputs)



    num_classes1 = cm.shape[0]

    # Generate class labels based on the number of classes
    classes = [f'Class {i}' for i in range(num_classes1)]

    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}

    # Calculate the accuracy for each one of your classes
    for idx, cls in enumerate(classes):
        # True negatives are all the samples that are not your current GT class (not the current row) 
        # and were not predicted as the current class (not the current column)
        true_n = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        
        # True positives are all the samples of your current GT class that were predicted as such
        true_p = cm[idx, idx]
        
        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (true_p + true_n) / np.sum(cm)

    # Calculate class-wise precision and recall
    for i in range(len(cm)):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        classwise_precision.append(precision)
        classwise_recall.append(recall)
    accperclass_str = np.array2string(accperclass, separator=', ')

    smallest_classes_indices = sorted(range(len(classwise_recall)), key=lambda i: classwise_recall[i])[:20]
    with open(txtpath, 'a') as file:
        print("Before writing")
        file.write("Class-wise Metrics (Recall and Precision):\n")
        for i, (rec, prec) in enumerate(zip(classwise_recall, classwise_precision)):
            file.write(f'Class {i} Recall: {rec:.2f} Precision: {prec:.2f}\n')
        file.write("5 Classes with Smallest Recall (Class Index and Recall Value):\n")
        for class_idx in smallest_classes_indices:
            recall_value = classwise_recall[class_idx]
            file.write(f'Class {class_idx} Recall: {recall_value:.2f}\n')
        print("After writing")    

    # Write class-wise precision to a file

    return (losses.avg, top1.avg, accperclass)

def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x/c)) - d


def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    return list(class_num_list)

def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param=ema_param.float()
            param=param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def logcm(cm,name,path):
    with open(path, 'a') as file:        
        file.write(name+'\n')
        for row in cm:
            formatted_row = [format(value, '.2f') for value in row]
            formatted_string = ' '.join(formatted_row)
            file.write(formatted_string+'\n')


'''
def clsloss(input_tensor,target_tensor,num_class,mask = None,criterion = 0):
    classwise_loss = []
    num_classes = num_class
    for i in range(num_classes):
        class_mask = (target_tensor == i)
        if class_mask.any():
            if criterion == 1:
                if mask:
                    class_loss = F.cross_entropy(input_tensor[class_mask]*mask[class_mask], target_tensor[class_mask]*mask[class_mask],reduction = 'sum')
                    classwise_loss.append(class_loss.item())
                else:
                    class_loss = F.cross_entropy(input_tensor[class_mask], target_tensor[class_mask],reduction = 'sum')
                    classwise_loss.append(class_loss.item())
            elif criterion == 2:
                if mask:
                    class_loss = torch.sum(input_tensor[class_mask]*target_tensor[class_mask]*mask[class_mask],dim=0)
                    classwise_loss.append(class_loss.item())
                else:
                    class_loss = torch.sum(input_tensor[class_mask]* target_tensor[class_mask],dim=0)
                    classwise_loss.append(class_loss.item())
            else:
                if mask:
                    class_loss = torch.sum(-F.log_softmax(input_tensor[class_mask],dim=1) * target_tensor[class_mask] * mask[class_mask],dim=0)
                    classwise_loss.append(class_loss.item())
                else:
                    class_loss = torch.sum(-F.log_softmax(input_tensor[class_mask],dim=1) * target_tensor[class_mask],dim=0)
                    classwise_loss.append(class_loss.item())
        else:
            classwise_loss.append(0.0)
        return torch.tensor(classwise_loss)            
'''
'''
def clsloss(input_tensor,target_tensor,num_class,mask = None):
    classwise_loss = []
    num_classes = num_class
    for i in range(num_classes):
        class_mask1 = (target_tensor == i)
        class_mask2 = (target_tensor[:, i] == 1)
        if class_mask1.any() or class_mask2.any:
            if mask is not None:
                print('i',i,'\n')
                print('target tensor\n',target_tensor,'\n')
                print('len of class  mask',len(class_mask),'len of mask',len(mask))
                print('class_mask\n',class_mask,'\n')
                print('mask\n',mask,'\n')
                print('try',mask[class_mask])
                print('attempt success')
                class_loss = torch.sum(F.cross_entropy(input_tensor[class_mask], target_tensor[class_mask],reduction = 'none')*mask[class_mask])
                classwise_loss.append(class_loss.item())
            else:
                class_loss = F.cross_entropy(input_tensor[class_mask], target_tensor[class_mask],reduction = 'sum')
                classwise_loss.append(class_loss.item())
        else:
            classwise_loss.append(0.0)

        return torch.tensor(classwise_loss)
'''
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

'''
def cal_fn_pair(cls_rep_x,cls_center_x,num_classes,name):
    #here we store furthest distance and nearest distance
        fdist = torch.full((num_classes,num_classes),-1)
        ndist = torch.full((num_classes,num_classes),-1)
        #here we store furthest sample and nearest sample
        ftensor_x = torch.zeros(num_classes,num_classes,128)
        ntensor_x = torch.zeros(num_classes,num_classes,128)   

        furest_smp_dst = torch.full((num_classes,), -1)

        for i, (index, representations_i) in enumerate(cls_rep_x.items()):
            for idx in range(len(cls_rep_x[i])):
                #update furthest distance between a sample and its class
                if spec_dist(cls_rep_x[i][idx],cls_center_x[i])>furest_smp_dst[i]:
                    furest_smp_dst[i] = spec_dist(cls_rep_x[i][idx],cls_center_x[i])
                    #print('furest_smp_dst[i]',furest_smp_dst[i])
                #loop through every class to find most distant and nearest pair  
                for j in range(len(cls_rep_x)):
                    #label_j, representations_j = list(class_representations.items())[j]
                    class_j_center = cls_center_x[j]
                    dis = spec_dist(cls_rep_x[i][idx],class_j_center)
                    if i==j:
                        continue
                    if fdist[i][j] == -1:
                        fdist[i][j] == dis
                        ftensor_x[i][j] = cls_rep_x[i][idx]
                    if ndist[i][j] == -1:
                        ndist[i][j] == dis
                        ntensor_x[i][j] = cls_rep_x[i][idx]
                    if fdist[i][j] < dis:
                        fdist[i][j] = dis
                        ftensor_x[i][j] = cls_rep_x[i][idx]
                    if ndist[i][j] > dis:
                        ndist[i][j] = dis
                        ntensor_x[i][j] = cls_rep_x[i][idx]
        #print('len of ftensor_x[0][0]',len(ftensor_x[0][0]),'ftensor_x[0][0]\n',ftensor_x[0][0])
        #print('len of ntensor_x[0][0]',len(ntensor_x[0][0]),'ntensor_x[0][0]\n',ntensor_x[0][0])
        #here i need to add calculating nearest dist and furthest dist using these pairs 
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if i==j:
                    continue
                dist_f[i][j] = spec_dist(ftensor_x[i][j],ftensor_x[j][i])
                dist_n[i][j] = spec_dist(ntensor_x[i][j],ntensor_x[j][i])
        #maybe i can modify here to specify the situation about overlapping

        with open(txtpath, 'a') as file:
            
            file.write(name+"\n")
            file.write('Most farthest distance between i and j'+'\n')
            for row in dist_f:
                row = row.cpu().detach().numpy()
                file.write(" ".join(map(str, row)) + "\n")
            file.write('Most nearest distance between i and j'+'\n')
            for row in dist_n:
                row = row.cpu().detach().numpy()
                file.write(" ".join(map(str, row)) + "\n")
            file.write('Distance between most distant sample and class center' + '\n')
            #file.write("{}\n".format(furest_smp_dst))
            fsd = furest_smp_dst.cpu().detach().numpy()
            file.write(" ".join(map(str, fsd)) + "\n")
'''
def cal_fn_pair(cls_rep_x,cls_center_x,num_classes,name):
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
                spec_dist_i = spec_dist(rep, cls_center_x[i])
                furest_smp_dst[i] = max(furest_smp_dst[i], spec_dist_i)
                
                #loop through every class to find most distant and nearest pair  
                for j ,  representations_j in cls_rep_x.items():
                    #label_j, representations_j = list(class_representations.items())[j]
                    
                    if i==j:
                        continue
                    class_j_center = cls_center_x[j]
                    dist_ij = spec_dist(cls_rep_x[i][idx],class_j_center)
                    if fdist[i][j] == -1 or fdist[i][j] < dist_ij:
                        fdist[i][j] = dist_ij
                        ftensor_x[i][j] = rep
                    if ndist[i][j] == -1 or ndist[i][j] > dist_ij:
                        ndist[i][j] = dist_ij
                        ntensor_x[i][j] = rep
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i, rep_i in enumerate(ftensor_x):
            for j, rep_j in enumerate(rep_i):
                if i == j:
                    continue
                dist_f[i][j] = spec_dist(rep_j, ftensor_x[j][i])
                dist_n[i][j] = spec_dist(ntensor_x[i][j], ntensor_x[j][i])
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

def cal_fn_pair1(cls_rep_x,cls_center_x,num_classes,name):
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
                spec_dist_i = spec_dist(rep, cls_center_x[i])
                furest_smp_dst[i] = max(furest_smp_dst[i], spec_dist_i)
                
                #loop through every class to find most distant and nearest pair  
                for j ,  representations_j in cls_rep_x.items(): #100
                    #label_j, representations_j = list(class_representations.items())[j]
                    
                    if i==j:
                        continue
                    class_j_center = cls_center_x[j]
                    dist_ij = spec_dist(cls_rep_x[i][idx],class_j_center)
                    if fdist[i][j] == -1 or fdist[i][j] < dist_ij:
                        fdist[i][j] = dist_ij
                        ftensor_x[i][j] = rep
                    if ndist[i][j] == -1 or ndist[i][j] > dist_ij:
                        ndist[i][j] = dist_ij
                        ntensor_x[i][j] = rep
        '''
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(len(cls_rep_x)):
                #d_cls_rep_x = cls_rep_x[i].detach()
                #d_cls_center_x = cls_center_x[i].detach()
                f1 = executor.submit(tmp,cls_rep_x,cls_center_x,fdist[i],ndist[i],i,num_classes)
                ftensor_x[i],ntensor_x[i],fdist[i],ndist[i] ,furest_smp_dst[i]= f1.result()               
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i, rep_i in enumerate(ftensor_x):
            for j, rep_j in enumerate(rep_i):
                if i == j:
                    continue
                dist_f[i][j] = spec_dist(rep_j, ftensor_x[j][i])
                dist_n[i][j] = spec_dist(ntensor_x[i][j], ntensor_x[j][i])
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

def cal_fn_pair2(cls_rep_x,cls_center_x,num_classes,name):
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
                
                f1 = executor.submit(tmp,cls_rep_x,cls_center_x,fdist[i],ndist[i],i,num_classes)
                ftensor_x[i],ntensor_x[i],fdist[i],ndist[i] ,furest_smp_dst[i]= f1.result()               
        '''
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(10)
        
        for j in range(int(num_class/10)):
            pool_list = []
            for i in range(10*j,10*j+10):
                res = pool.apply_async(tmp,args =  (cls_rep_x,cls_center_x,fdist[i],ndist[i],i,num_classes))
                pool_list.append(res)
            pool.close()
            pool.join()
            for res1 in pool_list:
                ftensor_x[i],ntensor_x[i],fdist[i],ndist[i] ,furest_smp_dst[i]= res1.get()               
            
                        
        dist_f= torch.zeros(num_classes,num_classes)
        dist_n = torch.zeros(num_classes,num_classes)
        for i, rep_i in enumerate(ftensor_x):
            for j, rep_j in enumerate(rep_i):
                if i == j:
                    continue
                dist_f[i][j] = spec_dist(rep_j, ftensor_x[j][i])
                dist_n[i][j] = spec_dist(ntensor_x[i][j], ntensor_x[j][i])
        

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

def cal_fn_pair3(cls_rep_x,cls_center_x,num_classes,name):
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
                
def tmp(cls_rep_x,cls_center_x,ret3_o,ret4_o,i,num_class):
            ret1 = torch.zeros(num_class,128)#furthest
            ret2 = torch.zeros(num_class,128)#nearest
            ret3 = ret3_o#fdist[i]
            ret4 = ret4_o#ndist[i]
            max_dist = -1
            representations_i = cls_rep_x[i]#100
            for idx,rep in enumerate(representations_i):#num of sample in this cls
                #update furthest distance between a sample and its class
                #rep = rep.detach()
                #class_i_center = cls_center_x[i].detach()
                class_i_center = cls_center_x[i]
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

def cal_cent_dist(cls_center_x,spec_dist,name):
    dist = torch.zeros(len(cls_center_x),len(cls_center_x))
    for i in range(len(cls_center_x)):
        for j in range(i,len(cls_center_x)):
            dist[i][j] = spec_dist(cls_center_x[i],cls_center_x[j])
            
    with open(txtpath, 'a') as file:           
            file.write(name +"\n")
            file.write('Distance between class center i and class center j'+'\n')
            for row in dist:
                row = row.cpu().detach().numpy()
                file.write(" ".join(map(str, row)) + "\n")

if __name__ == '__main__':
    main()