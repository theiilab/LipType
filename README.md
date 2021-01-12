# LipType

This is a Tensorflow implementation of the method described in the paper 'LipType: A Silent Speech Recognizer Augmented with an Independent Repair Model' by Laxmi Pandey and [Ahmed Sabbir Arif] (https://www.asarif.com/).

![Repaired LipType](repaired_liptype.gif)


## Dependencies
* Keras 2.0+
* Tensorflow 1.0+
* PIP (for package installation)
* editdistance==0.3.1
* h5py==2.6.0
* matplotlib==2.0.0
* numpy==1.12.1
* python-dateutil==2.6.0
* scipy==0.19.0
* Pillow==4.1.0
* tensorflow-gpu==1.0.1
* Theano==0.9.0
* nltk==3.2.2
* sk-video==1.1.7
* dlib==19.4.0
* scikit-learn 0.24.0
* NLTK

## Description
The contribution of this work is threefold: <br/> 
**LipType**
```
* Source: LipNet with some modifications in model2.py [ https://github.com/rizkiarm/LipNet ]
* In order to reduce computation and improve accuracy, we replaced a deep 3D CNN with a combination of a shallow 3D-CNN (1-layer) and a deep 2D-CNN (34-layer ResNet) integrated with squeeze and excitation (SE) blocks (SE-ResNet) (model2.py).
* Usage: refer LipType/README.md
```
**Pre-processing: Low-light enhancement**
```
* Source: GLADNet with some modifications in model.py [ https://github.com/weichen582/GLADNet ]
* In order to reduce computation, we changed the GLADNet network dimension from five down- and five up-sampling blocks to three down- and three up-sampling blocks.
* Changed the L1 loss to MSSSIM-L1 loss for improved performance (model.py).
* Usage: refer preprocessing/README.md
```
**Post-processing: Error Correction**
```
* In order to correct to predict and automatically correct potential recognition errors by LipType, we developed a 5-layer encoder-decoder architecture [128 64 32 64 128], followed by a sequence decoder comprising of spell checker, bi-directional ngram LM and Euclidean distance based word correction.
* Usage: refer postprocessing/README.md
```


## References
Read the following paper for further details. If you use this code, please cite our work.
```
Laxmi Pandey, Ahmed Sabbir Arif. 2021. LipType: A Silent Speech Recognizer Augmented with an Independent Repair Model. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI 2021). ACM, New York, NY, USA, to appear.
```
To read about the social acceptability of the LipType, please refer to the following paper.
```
Laxmi Pandey, Khalad Hasan, Ahmed Sabbir Arif. 2021. Acceptability of Speech and Silent Speech Input Methods in Private and Public. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI 2021). ACM, New York, NY, USA, to appear.
```
