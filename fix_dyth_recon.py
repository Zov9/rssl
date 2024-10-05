# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# First version of Supervised Contrastive loss
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import statistics
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

from losses import *
from uts import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
parser.add_argument('--lr1', '--learning-rate1', default=0.03, type=float,
                    metavar='LR', help='initial learning rate for sgd')
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
parser.add_argument('--date',type = str)
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
parser.add_argument('--closstemp', type=float, default=0.5, help='Temperature for contrastive learning')
parser.add_argument('--distance', type=float, default=0.5, help='Temperature for contrastive learning  supconloss1')
parser.add_argument('--wk', type=int, default=20, help='Worst classes number to work on')
parser.add_argument('--tempt', type=str, default='', help='attempt for txtpath')
parser.add_argument('--tau2', default=2, type=float,
                        help='tau for head2 balanced CE loss')
parser.add_argument('--lam', default=1, type=float,
                        help='coeffcient of closs')
parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
parser.add_argument('--conu', default=False,
                        help='Contrastive Loss on datau use or not')
parser.add_argument('--cl12',type = float, default=1.0,
                        help='coffecient of adjustment_l12')
parser.add_argument('--lam1', default=0.05, type=float,
                        help='coeffcient of OECC loss')
parser.add_argument('--use-la', default=False,
                        help='Contrastive Loss on datau use or not')
parser.add_argument('--comb', default=False,
                        help='whether to use lx lu lxb lub abcloss together ')
parser.add_argument('--use-sgd', default=False,
                        help='whether to use sgd as optimizer ')
parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
parser.add_argument('--wkthreshold', default=0.9, type=float,
                        help='pseudo label threshold for wk')
parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
parser.add_argument('--dismod', default='', type=str,
                        help='mode for calculate furest sample distance')
parser.add_argument('--diskey', default=1, type=int,
                        help='key for calculate furest sample distance')
parser.add_argument('--txtp', default='', type=str,
                        help='txtpath exact dir')
parser.add_argument('--weakth', default=0.95, type=float,
                        help='dynamic threshold for worst class')         
parser.add_argument('--lower_bound', default=0.55, type=float,
                        help='dynamic threshold for worst class')      
parser.add_argument('--higher_bound', default=0.7, type=float,
                        help='dynamic threshold for worst class')   
parser.add_argument('--usedyth', default=True,
                        help='whether to use dynamic threshold')                     
args = parser.parse_args()


#txtpath = "/data/lipeng/ABC/txt/try100_0502.txt"


state = {k: v for k, v in args._get_kwargs()}
if args.dataset=='cifar10':
    import dataset.fix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    dsstr = 'cf10'
    num_class = 10
elif args.dataset=='svhn':
    import dataset.fix_svhn as dataset
    print(f'==> Preparing imbalanced SVHN')
    dsstr = 'svhn'
    num_class = 10
elif args.dataset=='cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    dsstr = 'cf100'
    num_class = 100

args.out = './results/' + dsstr + '_' + args.date + 't' + args.tempt
args.txtp = '/data/lipeng/ABC/txt/' + dsstr + '_' + args.date + 't'
print('args.out is =====>',args.out)
print('args.txtp is =====>', args.txtp)
txtpath = args.txtp+args.tempt+'.txt'
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
    global twk

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

    args.py_con = compute_py(labeled_trainloader, args)
    args.adjustment_l2 = compute_adjustment_by_py(args.py_con, args.tau2, args)
    args.adjustment_l12 = compute_adjustment_by_py(args.py_con, args.tau2, args)

    

    model, params = create_model()
    ema_model,  _ = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    if(args.use_sgd == False):
        optimizer = optim.Adam(params, lr=args.lr)
        print('Adam')
    else:
        optimizer = optim.SGD(params, lr=args.lr1,
                          momentum=0.9, nesterov=args.nesterov)
        print('SGD')
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, 250000)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    worst_k = []
    info_pairs = []
    N = len(unlabeled_trainloader.dataset.indices)
    learning_status = [-1] * N

    # Resume
    title = 'ABCfix-' + args.dataset
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        worst_k = checkpoint['worst_k']
        info_pairs = checkpoint['info_pairs']
        learning_status = checkpoint['l_status']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
        
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'abcloss','Train Loss X b','Train Loss U b','Train Loss cl','Test Loss', 'Test Acc.'])

    #==================
    #unlabeled_trainloader
    
    #==================
    
    twk = []
    

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        '''
        if epoch>100:
            args.py_con = compute_py1(labeled_trainloader, worst_k) #flexmatch这里应该改一下 改成compute_py
            args.adjustment_l12 = compute_adjustment_by_py(args.py_con, args.tau2, args)
        '''    
        # Training part
        train_loss, train_loss_x, train_loss_u, abcloss, train_loss_x_b, train_loss_u_b, train_loss_cl, worst_k, info_pairs = train(labeled_trainloader,
                                                                                                unlabeled_trainloader,
                                                                                                model, optimizer,
                                                                                                ema_optimizer,
                                                                                                train_criterion,
                                                                                                epoch,ir2,scheduler,worst_k,info_pairs,
                                                                                                learning_status=learning_status,N = N)

        test_loss, test_acc, testclassacc,twk = validate(test_loader, ema_model, criterion, mode='Test Stats ',epoch = epoch)
        print('\ntrue wk next epoch',twk,'\n')
        if args.dataset == 'cifar10':
            print("each class accuracy test", testclassacc, testclassacc.mean(),testclassacc[:5].mean(),testclassacc[5:].mean())
        elif args.dataset == 'svhn':
            print("each class accuracy test", testclassacc, testclassacc.mean(), testclassacc[:5].mean(),testclassacc[5:].mean())
        elif args.dataset == 'cifar100':
            print("each class accuracy test", testclassacc, testclassacc.mean(), testclassacc[:50].mean(),testclassacc[50:].mean())

        logger.append([train_loss, train_loss_x, train_loss_u,abcloss, train_loss_x_b, train_loss_u_b, train_loss_cl , test_loss, test_acc])

        # Save models
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),

                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'worst_k':worst_k,
                'info_pairs':info_pairs,
                'l_status':learning_status

            }, epoch + 1)

    logger.close()






def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,ir2,scheduler, worst_k,info_pairs,learning_status,N):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x_b = AverageMeter()
    losses_u_b = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_abc = AverageMeter()
    losses_cl = AverageMeter()

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
    counter = Counter()
    counter1 = Counter()


    dy_threshold =  torch.full((num_class,), 0.95).cuda()
    dict_from_pairs = {index: value for index, value in info_pairs}   

    for item in worst_k:
            if info_pairs is not None and epoch>100 and args.usedyth == True :  #add epoch > 100 after checked no bug
                biggest_value = max(dict_from_pairs.values()) 
                #mean_value = statistics.mean(dict_from_pairs.values())
                mean_value = torch.mean(torch.tensor(list(dict_from_pairs.values())), dim=0)
                if not math.isnan(dict_from_pairs[item]):
                    dy_threshold[item] = args.lower_bound + (mean_value/dict_from_pairs[item])*(args.higher_bound-args.lower_bound)
                    #make sure it is in area (0.5,0.95)
                    dy_threshold[item] = min(dy_threshold[item],0.95)
                    dy_threshold[item] = max(dy_threshold[item],0.5)
                else:
                    dy_threshold[item] =  args.weakth
            else:      
                if epoch>100:
                    dy_threshold[item] = args.weakth
    print('dynamic_thresholds',dy_threshold) 

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
        counter1 = counter


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

            N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio,args.imbalancetype)
            result_arr = [i/sum(N_SAMPLES_PER_CLASS) for i in N_SAMPLES_PER_CLASS]

            
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
        '''
        if torch.isnan(q).any():
            print(f"NaN values detected in predictions for batch {batch_idx}. Skipping this batch.")
            continue
        '''
        q2 = model(inputs_u2)
        q3 = model(inputs_u3)

        max_p, p_hat = torch.max(targets_u2, dim=1)
        #print("len(max_p)",len(max_p),"max_p[0]",max_p[0])
        p_hat_mx_tmp  = p_hat
        
        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
           
        selected_thresholds = dy_threshold[p_hat_mx_tmp]
        #print('selected_thresholds',selected_thresholds)

        select_mask = max_p.ge(selected_thresholds)
        #select_mask1 = max_p.ge(0.95)
        #print('select mask',select_mask)
        #print('select mask1',select_mask1)
        smask = max_p.ge(0.9)
        #smask 这里是用在closs里面的 到底是用这个还是 select_mask 后面可以再看
        #la没效果 怀疑是mask的问题 edited on 24-09-26
        org_mask = select_mask
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)
        all_rtargets = torch.cat([targets_x2,targets_su2,targets_su2],dim=0)

        logits_x=model.classify(q)
        #outputs_u= model.classify(q1)  this is written elsewhere here to note
        logits_u1=model.classify(q2)
        logits_u2=model.classify(q3)
        logits_u = torch.cat([logits_u1,logits_u2],dim=0)

        maskforbalance = torch.bernoulli(torch.sum(targets_x2 * torch.tensor(ir2).cuda(0), dim=1).detach())

        logit = model.classify(q)#labeled sample
        logitu1 = model.classify(q1)#weak aug unlabel
        logitu2 = model.classify(q2)
        logitu3 = model.classify(q3)
        logitu23 = torch.cat([logitu2,logitu3],dim=0)

        logits = F.softmax(logit)#labeled sample
        logitsu1 = F.softmax(logitu1)#weak aug unlabel

        p1 = F.softmax(logits_x.detach())
        p2 = F.softmax(outputs_u.detach() - args.cl12 * args.adjustment_l12)
        p3 = F.softmax(logits_u1.detach())
        p4 = F.softmax(logits_u2.detach())
        p34 = F.softmax(logits_u.detach())

        p1b = F.softmax(logit.detach())
        p2b = F.softmax(outputs_u.detach())
        p3b = F.softmax(logitu2.detach())
        p4b = F.softmax(logitu3.detach())
        p34b = F.softmax(logitu23.detach())

        m1,t1 = torch.max(p1,dim =1)
        m2,t2 = torch.max(p2,dim =1)
        m3,t3 = torch.max(p3,dim =1)
        m4,t4 = torch.max(p4,dim =1)
        m34,t34 = torch.max(p34,dim =1)
        t2twice = torch.cat([t2,t2],dim=0).cuda()

        m1b,t1b = torch.max(p1b,dim =1)
        m2b,t2b = torch.max(p2b,dim =1)
        m3b,t3b = torch.max(p3b,dim =1)
        m4b,t4b = torch.max(p4b,dim =1)
        m34b,t34b = torch.max(p34b,dim =1)

        msk1w = torch.tensor([index in worst_k for index in t1], dtype=torch.bool)
        msk2w = torch.tensor([index in worst_k for index in t2], dtype=torch.bool)
        msk3w = torch.tensor([index in worst_k for index in t3], dtype=torch.bool)
        msk4w = torch.tensor([index in worst_k for index in t4], dtype=torch.bool)
        msk34w = torch.tensor([index in worst_k for index in t34], dtype=torch.bool)

        msk1bw = torch.tensor([index in worst_k for index in t1b], dtype=torch.bool)
        msk2bw = torch.tensor([index in worst_k for index in t2b], dtype=torch.bool)
        msk3bw = torch.tensor([index in worst_k for index in t3b], dtype=torch.bool)
        msk4bw = torch.tensor([index in worst_k for index in t4b], dtype=torch.bool)
        msk34bw = torch.tensor([index in worst_k for index in t34b], dtype=torch.bool)

        msk1s = m1.ge(args.wkthreshold)
        msk2s = m2.ge(args.wkthreshold)
        msk3s = m3.ge(args.wkthreshold)
        msk4s = m4.ge(args.wkthreshold)
        msk34s = m34.ge(args.wkthreshold)
        

        msk1sb = m1b.ge(args.wkthreshold)
        msk2sb = m2b.ge(args.wkthreshold)
        msk3sb = m3b.ge(args.wkthreshold)
        msk4sb = m4b.ge(args.wkthreshold)
        msk34sb = m34b.ge(args.wkthreshold)

        msk1 = m1.ge(args.threshold)
        msk2 = m2.ge(args.threshold)
        msk3 = m3.ge(args.threshold)
        msk4 = m4.ge(args.threshold)
        msk34 = m34.ge(args.threshold)

        msk1b = m1b.ge(args.threshold)
        msk2b = m2b.ge(args.threshold)
        msk3b = m3b.ge(args.threshold)
        msk4b = m4b.ge(args.threshold)
        msk34b = m34b.ge(args.threshold)


        msk2zz = msk2w.cuda() & msk2s.cuda()
        msk2zzb = msk2bw.cuda() & msk2sb.cuda()
        msk2z = msk2.cuda() 
        msk2zb = msk2b.cuda() 
        finalmask = msk2zz+msk2zzb+msk2z+msk2zb
        fnmask2 = torch.cat([finalmask,finalmask],dim=0).cuda()

        with torch.no_grad():
            max_prob, hard_label = torch.max(logitsu1, dim=1)
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
        smask2 = max_p2.ge(0.9)
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
        
        
            

        


        #if len(worstk)>0  then closs =  /    loss+= closs
        #print(' Worsk K in input argment:',worst_k)

        abcloss = -torch.mean(maskforbalance * torch.sum(torch.log(logits) * targets_x2.cuda(0), dim=1))
        abcloss1 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu2) * logitsu1.cuda(0).detach(), dim=1))

        abcloss2 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu3) * logitsu1.cuda(0).detach(), dim=1))

        totalabcloss=abcloss+abcloss1+abcloss2
        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        #print('!!!select mask!!!')  #64*2 len = 128
        #print(select_mask)
        #Lx_b = F.cross_entropy(logits_x + args.adjustment_l2, all_targets[:batch_size], reduction='mean')
        #Lu_b = F.cross_entropy(logits_u + args.adjustment_l2, all_targets[batch_size:], reduction='mean')
        '''
        Lu_b = F.cross_entropy(logits_u + args.adjustment_l2, all_targets[batch_size:], reduction='none')
        
        Lu_b = (Lu_b*select_mask).mean()
        '''

        #0224
        Lx_b = F.cross_entropy(logits_x + args.adjustment_l2, all_targets[:batch_size], reduction='mean')

        Lu_b = (F.cross_entropy(logits_u, t2twice,
                                    reduction='none') * select_mask).mean()

        #loss = Lx + Lu+totalabcloss
        #if args.use_la == True and epoch>100:
        if args.use_la == True :
            loss = Lx_b + Lu_b #+totalabcloss
            print('use-la logit adjustment')
        #elif args.comb == True and epoch>100:
        #    loss = Lx + Lu + Lx_b + Lu_b #+totalabcloss
        else:
            loss = Lx + Lu #+totalabcloss
            print('no logit adjustment')
        #loss = Lx + Lu_b+totalabcloss
        #criterionu = CSLoss2(temperature=args.closstemp)
        #criterionu = SupConLoss2(temperature=args.closstemp)
        criterionu = SupConLoss3(temperature=args.closstemp,distance = args.distance)
        criterionx = SupConLoss6(temperature=args.closstemp,distance=args.distance)
        #criterionu = SupConLoss3(temperature=args.closstemp,distance=args.distance)
        #criterion1 = criterion1.cuda()
        #print('inputs_x[0][:5]',inputs_x[0][:5])
        #print('q[0][:5]',q[0][:5])
        #print('clossu logit[0][:5]',logit[0][:5])
        #print('clossu outputs_u[0][:5]',outputs_u[0][:5])
        #clossu = criterionu(smask,worst_k,logit,logitu1,all_targets[:batch_size],targets_u)
        #clossu = criterionu(smask,worst_k,logit,logitu1,all_targets[:batch_size],hard_label)#supconloss2
        #clossu = criterionu(smask,worst_k,q,q1,all_targets[:batch_size],hard_label) #0201 for supconloss2
        #clossu = criterionu(smask,logitu1,hard_label,worst_k)#csloss2
        #clossu = criterionu(smask,q1,hard_label,worst_k)

        #0208 clossu and clossx below
        clossu = criterionu(smask,worst_k,logits_x,outputs_u,all_targets[:batch_size],targets_u)#supconloss2
        #clossu = criterionu(smask,worst_k,logits_x,outputs_u,all_targets[:batch_size],targets_su)
        clossx = criterionx(worst_k,logits_x,all_targets[:batch_size])

        #clossu = criterionu(smask,worst_k,q,q1,all_targets[:batch_size],targets_su)
        #clossu = criterionu(smask,q1,targets_su,worst_k)
        #clossx = criterionx(worst_k,logit,all_targets[:batch_size])

     
        
        if epoch>100 and args.conu == True:
             #loss = Lx_b+Lu_b+totalabcloss
            #if not torch.isnan(clossu).any():
                loss=loss+clossu #+clossx 
            
        print('Total loss',loss)


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
        losses_x_b.update(Lx_b.item(), inputs_x.size(0))
        losses_u_b.update(Lu_b.item(), inputs_x.size(0))
        losses_abc.update(abcloss.item(), inputs_x.size(0))
        losses_cl.update(clossu.item(), inputs_x.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.use_sgd:
            print('using sgd... updating sgd params')
            scheduler.step()
        ema_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}| Loss_m: {loss_m:.4f} | Loss_x_b: {loss_x_b:.4f}| Loss_u_b: {loss_u_b:.4f}| Loss_cl: {loss_cl:.4f}'.format(
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
                    loss_x_b=losses_x_b.avg,
                    loss_u_b=losses_u_b.avg,
                    loss_cl=losses_cl.avg,
                    )
        bar.next()
    bar.finish()

    #print('num of labeled tensors',num_alli)
    #print('num of unlabeled tensors',num_allu)
    #print('lxct',lxct)
    #print('luct',luct)
    #print('lurct',lurct)
    '''
    with open(txtpath,'a') as file:
            file.write('lxct\n')
            file.write(','.join(map(str, lxct))) 
            file.write('\nluct\n')
            file.write(','.join(map(str, luct))) 
            file.write('\nlurct\n')
            file.write(','.join(map(str, lurct)))  
            file.write('\n')
    '''  
            

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
    distx = cal_cent_dist(cls_center_x,spec_dist,'labeled center',txtpath)
    distur = cal_cent_dist(cls_center_ureal,spec_dist,'unlabeled real center',txtpath)
    distus = cal_cent_dist(cls_center_upsu_part,spec_dist,'unlabeled selected center',txtpath)
    distua = cal_cent_dist(cls_center_upsu_all,spec_dist,'unlabeled all center',txtpath)

        # Calculate the nearest and furthest samples for each pair of classes
    num_classes = num_class
    current_time = datetime.now()    

    print("555 Current Time:", current_time)
    
    fdx = cal_fn_pair3(cls_rep_x,cls_center_x,num_classes,'cls_rep_x',txtpath)

    current_time = datetime.now()
    print("558 Current Time:", current_time)
    
    fdur = cal_fn_pair3(cls_rep_ureal,cls_center_ureal,num_classes,'cls_rep_ureal',txtpath)    

    current_time = datetime.now()
    print("561 Current Time:", current_time)
  
    fdus = cal_fn_pair4(cls_rep_upsu_part,cls_center_upsu_part,num_classes,'cls_rep_upsu_part',txtpath,args.dismod,args.diskey)
   
    current_time = datetime.now()
    print("564 Current Time:", current_time)
                           
    fdua = cal_fn_pair3(cls_rep_upsu_all,cls_center_upsu_all,num_classes,'cls_rep_upsu_all',txtpath)
                
    current_time = datetime.now()
    print("567 Current Time:", current_time)

    worst_k,info_pairs = worstk(args.wk,distus,fdus)

        ############################################
        ############################################
               
    with open(txtpath,'a') as file:
            file.write("Epoch: "+str(epoch)+'\n')
            file.write('Worst k predicted\n') 
            file.write(str(worst_k))
            file.write('\n') 

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
  
    return (losses.avg, losses_x.avg, losses_u.avg, losses_abc.avg, losses_x_b.avg,  losses_u_b.avg, losses_cl.avg,  worst_k, info_pairs)

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
            #outputs2 = model.classify2(q)
            outputs2 = model.classify(q)
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

    return (losses.avg, top1.avg, accperclass,tuple(smallest_classes_indices))

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



if __name__ == '__main__':
    main()