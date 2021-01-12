# GLADNet with some modifications in model.py 
In order to reduce computation, we changed the GLADNet network dimension from five down- and five up-sampling blocks to three down- and three up-sampling blocks. We also changed the L1 loss to MSSSIM-L1 loss for improved performance. <br>
Source:  https://github.com/weichen582/GLADNet 


## Requirements ##
* Python
* Tensorflow >= 1.3.0
* numpy, PIL

## Testing ##
```
python main.py 
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.5 \                         # gpu memory usage
    --phase=test \
    --test_dir=/path/to/your/test/dir/ \
    --save_dir=/path/to/save/results/ \
```
## Training ##
```
python main.py
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.8 \                         # gpu memory usage
    --phase=train \
    --epoch=50 \                            # number of training epoches
    --batch_size=8 \
    --patch_size=384 \                      # size of training patches
    --base_lr=0.001 \                       # initial learning rate for adm
    --eval_every_epoch=5 \                  # evaluate and save checkpoints for every # epoches
    --checkpoint_dir=./checkpoint           # if it is not existed, automatically make dirs
    --sample_dir=./sample                   # dir for saving evaluation results during training
 ```
