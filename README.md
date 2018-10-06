# DenseVideoCaptioning

Tensorflow Implementation of the Paper [Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning](https://arxiv.org/abs/1804.00100) by Jingwen Wang *et al.* in *CVPR* 2018.

### Data Preparation

Please download annotation data and C3D features from the website [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/).

Please follow the script dataset/ActivityNet_Captions/preprocess/anchors/get_anchors.py to obtain clustered anchors and their pos/neg weights (for handling imbalance class problem). I already put the generated files in dataset/ActivityNet_Captions/preprocess/anchors/.

Please follow the script dataset/ActivityNet_Captions/preprocess/build_vocab.py to build word dictionary and to build train/val/test encoded sentence data.

### Hyper Parameters

The configuration (from my experiments) is given in opt.py, including model setup, training options, and testing options.

### Training

Train dense-captioning model using the script train.py.

First pre-train the proposal module for around 5 epochs. Set train_proposal=True and train_caption=False. Then train the whold dense-captioning model by setting train_proposal=True and train_caption=True

### Prediction

Follow the script test.py to make proposal predictions and to evaluate the predictions.

### Evaluation

Please note that the official evaluation metric has been [updated](https://github.com/ranjaykrishna/densevid_eval/commit/bbbd49d31a038acf2642f7ae158bb6b9da6937fc)(Line 194). In the paper, old metric is reported (but still, you can compare results from different methods, all CVPR-2018 papers report old metric).

### Results

The predicted results for val/test set can be found in results/.

### Dependencies

tensorflow==1.0.1

python==2.7.5

Other versions may also work.