#1219
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1219t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --resume /data/lipeng/ABC/cf100_1120/model_100.pth.tar
#tksps
CUDA_VISIBLE_DEVICES=0 python ABCfix.py --dataset cifar100 --out cf100_zz --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 

CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1221L2T05 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --resume /data/lipeng/ABC/cf100_1120/model_100.pth.tar

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1221L2T01 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --resume /data/lipeng/ABC/cf100_1120/model_100.pth.tar

CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1221L2T1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --resume /data/lipeng/ABC/cf100_1120/model_100.pth.tar

#1222
#1223
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1222t8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --tempt 8 --resume /data/lipeng/ABC/cf100_1120/model_100.pth.tar
#1224
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1222t14 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --tempt 14 
#1225
#1228
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1229Rt2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --tempt 2
#1230
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1230t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --tempt 1
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_1230t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.3 --tempt 2
#0101
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0101t4 --label_ratio 8 --imb_ratio 20 --num_max 150  --epochs 500 --closstemp 0.1  --tempt 4
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0101t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5  --tempt 6
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0101t7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1  --tempt 7
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0101t8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1  --tempt 8
#0104
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0104t11 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1  --tempt 11
#
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0105t32 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5  --tempt 32 --resume /data/lipeng/ABC/cf100_0105t32/checkpoint.pth.tar
#0107
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0107t32 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.5  --tempt 32  --resume /data/lipeng/ABC/cf100_0107t32/checkpoint.pth.tar
#0108
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0108t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --distance 0.5  --tempt 6  --resume /data/lipeng/ABC/cf100_0108t6/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0108t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --distance 0.5  --tempt 6  --resume /data/lipeng/ABC/cf100_0108t6/checkpoint.pth.tar

#0111
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0111t4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 0.1  --tempt 4
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0111t5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 0.5  --tempt 5
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0111t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 1  --tempt 6

#0113
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0111t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 1 --oe False  --tempt 6 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0111t6/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0113t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --resume /data/lipeng/ABC/cf100_0113t2/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0113t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 1 --oe True  --tempt 1 --conu False --use-la False --resume /data/lipeng/ABC/cf100_0113t1/checkpoint.pth.tar
#0114
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0114t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 0.1 --oe False  --tempt 1 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0114t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 0.5 --oe False  --tempt 2 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0114t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 3 --conu True --use-la True 

#0115
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 0.1 --oe False  --tempt 1 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --lam 0.1 --oe False  --tempt 2 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 0.1 --oe False  --tempt 3 --conu True --use-la True 

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 0.5 --oe False  --tempt 4 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0115t4/checkpoint.pth.tar  
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --lam 0.5 --oe False  --tempt 5 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0115t5/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 0.5 --oe False  --tempt 6 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0115t6/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 7 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0115t7/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --lam 1 --oe False  --tempt 8 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0115t8/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0115t9 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 1 --lam 1 --oe False  --tempt 9 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0115t9/checkpoint.pth.tar 

#0119
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0118t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 1 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0118t1/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0118t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 2 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0118t2/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0118t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 3 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0118t3/checkpoint.pth.tar 
#0125
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0113t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe True  --tempt 1 --conu False --use-la False --resume /data/lipeng/ABC/cf100_0113t1/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0113t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --resume /data/lipeng/ABC/cf100_0113t2/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=3 python train3.py --dataset cifar100 --num-max 150 --num-max-u 300 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 10 --imb-ratio-unlabel 10 --ema-u 0.99 --out out/cifar-100/N150_M300/0124 --resume /data/lipeng/clone/ACR/out/cifar-100/N150_M300/0124/checkpoint.pth.tar 
#0126
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0129t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 3 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0129t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 1 --conu True --use-la True 
#下面这个还没跑
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0126t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.5 --lam 1 --oe False  --tempt 2 --conu True --use-la True 

#0131 try
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t0 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 0 --conu True --use-la True

#0131
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 1 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t1/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 2 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t2/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 3 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t3/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 4 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t4/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 5 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t5/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 6 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t6/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 7 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t7/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 8 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t8/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0131t9 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 9 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0131t9/checkpoint.pth.tar 

#0131 tempt1-3 还没跑
conda activate torch1.12
cd ABC

#0208
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0208t1/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0208t2/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0208t3/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 7 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0208t7/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 8 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0208t8/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t9 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 9 --conu True --use-la True --resume /data/lipeng/ABC/cf100_0208t9/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 4 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 5 --conu True --use-la True 
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0208t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 6 --conu True --use-la True

conda activate torch1.12
cd ABC

#0218
#original abc
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0218t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 1 --conu False --use-la False 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0218t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 2 --conu False --use-la False
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0218t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 3 --conu False --use-la False
#1222t12  from  scratch     LA on lu lx  and supconloss3 tau 0.1 
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0218t4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 4 --conu True --use-la True
#cf100_1229R1.txt original LA on lx and lu from scratch closs on lu and lx,  lu supconloss2  lx supconloss6 tau = 0.5  distance = 0.5
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0218t5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 5 --conu True --use-la True
#12301 or 12302
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0218t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 6 --conu True --use-la True

##0222
#1222t12  from  scratch     LA on lu lx  and supconloss3 tau 0.1 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0222t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --lam 1 --oe False  --tempt 1 --conu True --use-la True
#cf100_1229R1.txt original LA on lx and lu from scratch closs on lu and lx,  lu supconloss2  lx supconloss6 tau = 0.5  distance = 0.5
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0222t2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.5 --distance 0.5 --lam 1 --oe False  --tempt 2 --conu True --use-la True
#12301 or 12302
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0222t3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu True --use-la True

CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0222t4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 4 --conu False --use-la True

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0222t5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 5 --conu False --use-la True

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0222t6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 6 --conu False --use-la True


#0225
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_4.py --dataset cifar100 --out cf100_0225t1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb True --resume /data/lipeng/ABC/cf100_0225t1/checkpoint.pth.tar



#0501
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t1 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod avg --diskey 2 --resume /data/lipeng/ABC/cf100_0501t1/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t2 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --comb False  --dismod avg --diskey 3 --resume /data/lipeng/ABC/cf100_0501t2/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t3 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu False --use-la False --comb False  --dismod avg --diskey 4 --resume /data/lipeng/ABC/cf100_0501t3/checkpoint.pth.tar
#0501 1458 以上已经运行了
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t4 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 4 --conu False --use-la False --comb False  --dismod avg --diskey 5 --resume /data/lipeng/ABC/cf100_0501t4/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t5 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 5 --conu False --use-la False --comb False  --dismod furst --diskey 1 --resume /data/lipeng/ABC/cf100_0501t5/checkpoint.pth.tar
#0501 1533 以上已经运行了
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t6 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 6 --conu False --use-la False --comb False  --dismod furst --diskey 2 --resume /data/lipeng/ABC/cf100_0501t6/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5.py --dataset cifar100 --out cf100_0501t7 --txtp /data/lipeng/ABC/txt/cf100_0501_1623t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 7 --conu False --use-la False --comb False  --dismod furst --diskey 3 --resume /data/lipeng/ABC/cf100_0501t7/checkpoint.pth.tar
#0501 1616 以上已经运行了

#0601 dynamic threshold / class-wise threshold
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0601try --txtp /data/lipeng/ABC/txt/cf100_0601tryt --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod avg --diskey 4 

#0602 dynamic threshold first exp
#base 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t1 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.95 --resume /data/lipeng/ABC/cf100_0602t1/checkpoint.pth.tar
#0.85
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t2 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.85 --resume /data/lipeng/ABC/cf100_0602t2/checkpoint.pth.tar
#0.8
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t3 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.8 --resume /data/lipeng/ABC/cf100_0602t3/checkpoint.pth.tar
#0.7
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t4 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 4 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.7 --resume /data/lipeng/ABC/cf100_0602t4/checkpoint.pth.tar


#0602 dynamic aug first exp
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_aug.py --dataset cifar100 --out cf100_0602_1503try --txtp /data/lipeng/ABC/txt/cf100_0602try2_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 4 --conu False --use-la False --comb False  --dismod avg --diskey 4 

#0606 dy th continue 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t5 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 5 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.75
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t6 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 6 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.65
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t7 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 7 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.6
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0602t8 --txtp /data/lipeng/ABC/txt/cf100_0602t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 8 --conu False --use-la False --comb False  --dismod avg --diskey 4 --weakth 0.55


#0610  use worst_k instead of twk
    #combine best worst class selection 之前挑最差类最好的是tempt7 准确率在70多 就用这个
CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0610t1 --txtp /data/lipeng/ABC/txt/cf100_0610t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55
    #TRY add lower bound higher bound, calculate threshold directly based on overlap_percentage output, applied after epoch 100
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0610try1 --txtp /data/lipeng/ABC/txt/cf100_0610try --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55
    #ahead try actual use add lower bound higher bound (here by default), calculate threshold directly based on overlap_percentage output, applied after epoch 100  actual use
    ***(abandoned) CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0610t2 --txtp /data/lipeng/ABC/txt/cf100_0610t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55
    
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0610_1343t2 --txtp /data/lipeng/ABC/txt/cf100_0610_1343t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55

#0706 
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0706t1 --txtp /data/lipeng/ABC/txt/cf100_0706t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.9 --lower_bound 0.5

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0706t2 --txtp /data/lipeng/ABC/txt/cf100_0706t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out cf100_0706t3 --txtp /data/lipeng/ABC/txt/cf100_0706t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth False --higher_bound 0.9 --lower_bound 0.5