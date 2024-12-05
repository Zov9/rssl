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
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0706t1 --txtp /data/lipeng/ABC/txt/cf100_0706t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.9 --lower_bound 0.5

CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0706t2 --txtp /data/lipeng/ABC/txt/cf100_0706t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55

CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0706t3 --txtp /data/lipeng/ABC/txt/cf100_0706t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth False --higher_bound 0.9 --lower_bound 0.5

#0713
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0713t1 --txtp /data/lipeng/ABC/txt/cf100_0713t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0713t2 --txtp /data/lipeng/ABC/txt/cf100_0713t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu True --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0713t3 --txtp /data/lipeng/ABC/txt/cf100_0713t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu False --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0713t4 --txtp /data/lipeng/ABC/txt/cf100_0713t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 4 --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
#0803
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0803t1 --txtp /data/lipeng/ABC/txt/cf100_0803t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 1 --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
CUDA_VISIBLE_DEVICES=0 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0803t2 --txtp /data/lipeng/ABC/txt/cf100_0803t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 2 --conu True --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
CUDA_VISIBLE_DEVICES=1 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0803t3 --txtp /data/lipeng/ABC/txt/cf100_0803t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 3 --conu False --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
CUDA_VISIBLE_DEVICES=2 python ABCfix_v6_5_dy_th.py --dataset cifar100 --out ./results/cf100_0803t4 --txtp /data/lipeng/ABC/txt/cf100_0803t --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1 --oe False  --tempt 4 --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55
    #sample command below  
    CUDA_VISIBLE_DEVICES=3 python ABCfix_v6_5_dy_th.py --dataset cifar100 --date 080x --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.55 --usedyth True --higher_bound 0.7 --lower_bound 0.55


#0909
#fix base
CUDA_VISIBLE_DEVICES=0 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.95 --usedyth False --higher_bound 0.7 --lower_bound 0.55 
#la only
CUDA_VISIBLE_DEVICES=0 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --usedyth False --higher_bound 0.7 --lower_bound 0.55 --resume /data/lipeng/ABC/results/cf100_0909t2/checkpoint.pth.tar
#dy th only
CUDA_VISIBLE_DEVICES=0 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.95 --usedyth True --higher_bound 0.7 --lower_bound 0.55 --resume /data/lipeng/ABC/results/cf100_0909t3/checkpoint.pth.tar
#dy th only + la
CUDA_VISIBLE_DEVICES=1 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --usedyth True --higher_bound 0.7 --lower_bound 0.55 --resume /data/lipeng/ABC/results/cf100_0909t4/checkpoint.pth.tar
#infonce + la
CUDA_VISIBLE_DEVICES=2 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.95 --usedyth False --higher_bound 0.7 --lower_bound 0.55 --resume /data/lipeng/ABC/results/cf100_0909t5/checkpoint.pth.tar
#comb + dy th
CUDA_VISIBLE_DEVICES=2 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.95 --usedyth True --higher_bound 0.7 --lower_bound 0.55 --resume /data/lipeng/ABC/results/cf100_0909t6/checkpoint.pth.tar

# 0926 补充 comb 是 lx lu + lxb lub 不是 infonce 所以重新跑后两组实验
#infonce + la
CUDA_VISIBLE_DEVICES=1 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --usedyth False --higher_bound 0.7 --lower_bound 0.55 
#comb + dy th
CUDA_VISIBLE_DEVICES=3 python fix_v6_5_dy_th.py --dataset cifar100 --date 0909 --tempt 8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --usedyth True --higher_bound 0.7 --lower_bound 0.55 



cd /data/lipeng/ABC
conda activate torch1.12

#0918  flexmatch  
    # flexmatch thre  for   1. all classes  2. worst 20 classes
    # la              for   2. worst 20 classes
    # infonce         for   2. worst 20 classes
#below are untested
#flex base 
CUDA_VISIBLE_DEVICES=1 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.95 --onlywst False
#flex base version 2  only worst class use dynamic threshold, others remain weakth
CUDA_VISIBLE_DEVICES=1 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb False  --dismod furst --diskey 3 --weakth 0.95 --onlywst True --resume /data/lipeng/ABC/results/cf100_0918t2/checkpoint.pth.tar
#la only
CUDA_VISIBLE_DEVICES=0 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --onlywst False
CUDA_VISIBLE_DEVICES=0 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --onlywst True
#infonce + la
CUDA_VISIBLE_DEVICES=2 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.95 --onlywst False
CUDA_VISIBLE_DEVICES=3 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False --comb True  --dismod furst --diskey 3 --weakth 0.95 --onlywst True

#0926 后两组 有问题
CUDA_VISIBLE_DEVICES=1 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --onlywst False
CUDA_VISIBLE_DEVICES=3 python flex_v6_5_dy_th.py --dataset cifar100 --date 0918 --tempt 8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True --comb False  --dismod furst --diskey 3 --weakth 0.95 --onlywst True

#1005
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la False   --dismod furst --diskey 3 --weakth 0.95 
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 1.0
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 2.0
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667

    #only la
python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 5_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True
    #la + infonce
python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True
python fix_dyth_recon.py --dataset cifar100 --date 1005 --tempt 6_2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --lbdcl 0.05

#1031
    #only la
python flex_dyth_recon.py --dataset cifar100 --date 1031 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True
    #la + infonce
python flex_dyth_recon.py --dataset cifar100 --date 1031 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True
#1113 fix efficiency
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1112 --tempt try1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu True --use-la True   --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True

#1118
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1118 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth False --use-la False --usecsl True --lbdcsl 1.5

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1118 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --use-la False --usecsl True --lbdcsl 1.5

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1118 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --use-la True --usecsl True --lbdcsl 1.5
#1119
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1119 --tempt try1 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --use-la True --usecsl True --lbdcsl 1.5 --wk 3

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1119 --tempt 1 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --use-la True --usecsl True --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1119 --tempt 2 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --use-la False --usecsl False --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1119 --tempt 3 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth True --use-la True --usecsl False --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1119 --tempt 0 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu False --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth False --use-la False --usecsl False --lbdcsl 1.5 --wk 3
# 上面 bool 型 argument 不生效 切换到下面
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 0_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 1_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3
# to be finish // finished
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3
    # la系数 0.4 最佳 对应于 2_3

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 3_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 4_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 1 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 5_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 0 --usecsl 1 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 6_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 1 --lbdcsl 1.5 --wk 3

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_1 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.2 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_2 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.3 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_3 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_4 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.1 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_5 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.5 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 2_6 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3


#1121 redo 1118 cost sensitive learning
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1121 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 1 --lbdcsl 1.5

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1121 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 0 --usecsl 1 --lbdcsl 1.5

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1121 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 1 --usecsl 1 --lbdcsl 1.5

#1122
python fix_dyth_recon.py --dataset cifar100 --date 1122 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 20
python ABCfix_dyth_recon.py --dataset cifar100 --date 1122 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 20 --resume /root/rssl/results/cf100_1122t1/checkpoint.pth.tar
python flex_dyth_recon.py --dataset cifar100 --date 1122 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 20

# 1124
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar10 --date 1124 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=2 python ABCfix_dyth_recon.py --dataset cifar10 --date 1124 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 --resume /root/rssl/results/cf10_1124t1/checkpoint.pth.tar 
CUDA_VISIBLE_DEVICES=0 python flex_dyth_recon.py --dataset cifar10 --date 1124 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 --resume /root/rssl/results/cf10_1124t3/checkpoint.pth.tar

#1125
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 3_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 3 --resume /root/rssl/results/cf10_1120t3_/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 4_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 1 --lbdcsl 1.5 --wk 3 --resume /root/rssl/results/cf10_1120t4_/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 5_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 0 --usecsl 1 --lbdcsl 1.5 --wk 3 --resume /root/rssl/results/cf10_1120t5_/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 6_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 1 --lbdcsl 1.5 --wk 3 --resume /root/rssl/results/cf10_1120t6_/checkpoint.pth.tar

    #other cof for csl
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1121 --tempt 1_2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 1 --lbdcsl 1.2 --resume /root/rssl/results/cf100_1121t1_2/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1121 --tempt 1_3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 1 --lbdcsl 1.3 --resume /root/rssl/results/cf100_1121t1_3/checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1121 --tempt 1_4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 0 --usecsl 1 --lbdcsl 1.4 --resume /root/rssl/results/cf100_1121t1_4/checkpoint.pth.tar
    #lower and upper bound for cifar10 for dyth
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1120 --tempt 1_2 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 --lower_bound 0.7 --higher_bound 0.95


#1126
    #1124 还是有问题 num_max 设置的不对 重新跑下
    #下面专供3090 实验结果再3090上
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 1127 --tempt 2 --label_ratio 2 --imb_ratio 100 --num_max 1000  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=0 python ABCfix_dyth_recon.py --dataset cifar10 --date 1127 --tempt 1 --label_ratio 2 --imb_ratio 100 --num_max 1000  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 
CUDA_VISIBLE_DEVICES=2 python flex_dyth_recon.py --dataset cifar10 --date 1127 --tempt 3 --label_ratio 2 --imb_ratio 100 --num_max 1000  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 


#1127
    # only dyth

python fix_dyth_recon.py --dataset cifar100 --date 1127 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 20
CUDA_VISIBLE_DEVICES=0 python ABCfix_dyth_recon.py --dataset cifar100 --date 1127 --tempt 1_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 20 
python flex_dyth_recon.py --dataset cifar100 --date 1127 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 20


python fix_dyth_recon.py --dataset cifar100 --date 1127 --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 20
CUDA_VISIBLE_DEVICES=3 python ABCfix_dyth_recon.py --dataset cifar100 --date 1127 --tempt 4_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 20 
python flex_dyth_recon.py --dataset cifar100 --date 1127 --tempt 6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.5 --wk 20

#1128 #1126还是有问题 nummax设置的不对
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar10 --date 1128 --tempt 2 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3
CUDA_VISIBLE_DEVICES=3 python ABCfix_dyth_recon.py --dataset cifar10 --date 1128 --tempt 1 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 
CUDA_VISIBLE_DEVICES=2 python flex_dyth_recon.py --dataset cifar10 --date 1128 --tempt 3 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 0 --usecsl 0 --lbdcsl 1.5 --wk 3 

#1130 测试不同LA tau 参数设置对 flexmatch 的影响
CUDA_VISIBLE_DEVICES=0 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.1 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=0 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.2 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=1 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.3 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=1 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=2 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.5 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=2 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=3 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20
CUDA_VISIBLE_DEVICES=3 python flex_dyth_recon.py --dataset cifar100 --date 1130 --tempt 8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.7 --usedyth 0 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20

# to be done maybe after modifing worst class select rep
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.1 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.2 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.3 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.5 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 6 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 7 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 8 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.7 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1

# below to be done by 2024 12 01 23:22 
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 1_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.1 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 2_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.2 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 3_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.3 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 4_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 5_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.5 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 6_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 7_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.6667 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 1201 --tempt 8_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.7 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 0
   
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar10 --date 1201 --tempt try1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.7 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1

# org mask or selected dynamic mask?
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 1 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 2 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 4 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 5 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 1_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 1 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 2_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 2 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 3_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 4_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 4 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 241204 --tempt 5_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 5 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
    #to be done by 1204 1147
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 1 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 1 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 2 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 2 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 3 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 4 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 4 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 5 --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 5 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 1_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 1 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 2_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 2 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 3_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 4_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 4 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar100 --date 241204_ --tempt 5_ --label_ratio 2 --imb_ratio 10 --num_max 150  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod avg --diskey 5 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
   #ahead is  not finished  try cifar10

CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 1 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 1 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 2 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 2 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 3 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 4 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 4 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1
CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 5 --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 5 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 1

CUDA_VISIBLE_DEVICES=2 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 1_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 1 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 2_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 2 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=3 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 3_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 3 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=0 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 4_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 4 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2
CUDA_VISIBLE_DEVICES=1 python fix_dyth_recon.py --dataset cifar10 --date 241205 --tempt 5_ --label_ratio 2 --imb_ratio 100 --num_max 1500  --epochs 500 --closstemp 0.1 --distance 0.1 --lam 1  --conu 0 --dismod furst --diskey 5 --weakth 0.95 --cl12 0.4 --usedyth 1 --use-la 1 --usecsl 0 --lbdcsl 1.4 --wk 20 --repmod 1 --omaskmod 2



