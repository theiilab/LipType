# Postprocessing
In order to correct to predict and automatically correct potential recognition errors by LipType, we developed a 5-layer encoder-decoder architecture [128 64 32 64 128], followed by a sequence decoder comprising of spell checker, bi-directional ngram LM and Euclidean distance based word correction.


## Requirements ##
1. Python
2. Tensorflow >= 1.3.0
3. numpy
4. scikit-learn 0.24.0
5. NLTK

## Training ##
python correction.py
    --phase=train \
    --epoch=50 \                           # number of training epoches
    --batch_size=128 \
    --base_lr=0.001 \                      # initial learning rate for adm
    --eval_every_epoch=5 \                 # evaluate and save checkpoints for every # epoches
    --checkpoint_dir=./checkpoint           # if it is not existed, automatically make dirs
    --sample_dir=./sample                   # dir for saving evaluation results during training

## Testing ##
python correction.py 
    --phase=test \
    --test_dir=/path/to/your/test/dir/ \
    --save_dir=/path/to/save/results/ \

