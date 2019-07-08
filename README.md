# DenseVideoCaptioning

Tensorflow Implementation of the Paper [Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning](https://arxiv.org/abs/1804.00100) by Jingwen Wang *et al.* in *CVPR* 2018.

![alt text](method.png)

### Citation

    @inproceedings{wang2018bidirectional,
      title={Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning},
      author={Wang, Jingwen and Jiang, Wenhao and Ma, Lin and Liu, Wei and Xu, Yong},
      booktitle={CVPR},
      year={2018}
    }

### Data Preparation

Please download annotation data and C3D features from the website [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/). The ActivityNet C3D features with stride of 64 frames (used in my paper) can be found in [https://drive.google.com/open?id=1UquwlUXibq-RERE8UO4_vSTf5IX67JhW](https://drive.google.com/open?id=1UquwlUXibq-RERE8UO4_vSTf5IX67JhW).

Please follow the script dataset/ActivityNet_Captions/preprocess/anchors/get_anchors.py to obtain clustered anchors and their pos/neg weights (for handling imbalance class problem). I already put the generated files in dataset/ActivityNet_Captions/preprocess/anchors/.

Please follow the script dataset/ActivityNet_Captions/preprocess/build_vocab.py to build word dictionary and to build train/val/test encoded sentence data.

### Hyper Parameters

The configuration (from my experiments) is given in opt.py, including model setup, training options, and testing options. You may want to set max_proposal_num=1000 if saving valiation time is not the first priority.

### Training

Train dense-captioning model using the script train.py.

First pre-train the proposal module (you may need to slightly modify the code to support batch size of 32, using batch size of 1 could lead to unsatisfactory performance). The pretrained proposal model can be found in https://drive.google.com/drive/folders/1IeKkuY3ApYe_QpFjarweRb2MTJKTCOLa. Then train the whole dense-captioning model by setting train_proposal=True and train_caption=True. To understand the proposal module, I refer you to the original [SST](http://openaccess.thecvf.com/content_cvpr_2017/papers/Buch_SST_Single-Stream_Temporal_CVPR_2017_paper.pdf) paper and also my tensorflow [implementation](https://github.com/JaywongWang/SST-Tensorflow) of SST.

### Prediction

Follow the script test.py to make proposal predictions and to evaluate the predictions. Use max_proposal_num=1000 to generate .json test file and then use script "python2 evaluate.py -s [json_file] -ppv 100" to evaluate the performance (the joint ranking requres to drop items that are less confident).

### Evaluation

Please note that the official evaluation metric has been [updated](https://github.com/ranjaykrishna/densevid_eval/commit/bbbd49d31a038acf2642f7ae158bb6b9da6937fc) (Line 194). In the paper, old metric is reported (but still, you can compare results from different methods, all CVPR-2018 papers report old metric).

### Pre-trained Model & Results

[Deprecated] The predicted results for val/test set can be found [here](https://drive.google.com/drive/folders/1KZfzoTV3qqtnzALwZgd5IU5BRkj69SZ8?usp=sharing). 

The pre-trained model and validation/test prediction can be found [here](https://drive.google.com/drive/folders/1qeH5r5XEabkcQDJ25unSCvEUziRleN80?usp=sharing). On validation set the model obtained 9.77 METEOR score using evaluate_old.py and 5.42 METEOR score using evaluate.py. On test set the model obtained 4.49 METEOR score returned by the ActivityNet server.

### Dependencies

tensorflow==1.0.1

python==2.7.5

Other versions may also work.

**NOTE:** 
1. Due to large file constraint, you may need to download data/paraphrase-en.gz [here](https://github.com/tylin/coco-caption/tree/3f0fe9b819c0ea881a56441e4de1146924a394eb/pycocoevalcap/meteor/data) and put it in densevid_eval-master/coco-caption/pycocoevalcap/meteor/data/.
