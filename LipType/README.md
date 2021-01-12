# # Source: LipNet with some modifications in model2.py [ https://github.com/rizkiarm/LipNet ]
# In order to reduce computation and improve accuracy, we replaced a deep 3D CNN with a combnation of a shallow 3D-CNN (1-layer) and a deep 2D-CNN (34-layer ResNet) integrated with squeeze and excitation (SE) blocks (SE-ResNet).

## Requirements ##
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

## Usage ##
To use the model, first you need to clone the repository:

cd LipNet/
pip install -e .

## Dataset ##
This model uses GRID corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)

## Training ##
./train unseen_speakers_curriculum [GPUs (optional)]

## Testing ##
./predict [path to weight] [path to video]
