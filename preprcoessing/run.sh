##Training##

python main.py
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.8 \                         # gpu memory usage
    --phase=train \
    --epoch=50 \                           # number of training epoches
    --batch_size=8 \
    --patch_size=384 \                       # size of training patches
    --base_lr=0.001 \                      # initial learning rate for adm
    --eval_every_epoch=5 \                 # evaluate and save checkpoints for every # epoches
    --checkpoint_dir=./checkpoint           # if it is not existed, automatically make dirs
    --sample_dir=./sample                   # dir for saving evaluation results during training

##Testing##

python main.py 
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.5 \                         # gpu memory usage
    --phase=test \
    --test_dir=/path/to/your/test/dir/ \
    --save_dir=/path/to/save/results/ \