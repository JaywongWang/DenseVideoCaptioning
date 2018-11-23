"""
Configurations: including model configuration and hyper parameter setting
"""

from collections import OrderedDict
import tensorflow as tf
import pickle as pkl
import numpy as np
import sys
import json
import time
import os

def default_options():

    options = OrderedDict()

    ### DATA
    options['feature_data_path'] = 'dataset/ActivityNet/features/sub_activitynet_v1-3_stride_64frame.c3d.hdf5' # download feature from ActivityNet website, and use a stride of 64 frames (shorten the unfolding steps for encoding LSTMs)
    options['localization_data_path'] = 'dataset/ActivityNet_Captions' 
    options['caption_data_root'] = 'dataset/ActivityNet_Captions/preprocess'
    options['vocab_file'] = os.path.join(options['caption_data_root'], 'word2id.json')
    options['vocab'] = json.load(open(options['vocab_file']))  # dictionary: word to word_id
    options['vocab_size'] = len(options['vocab'])   # number of words

    options['init_from'] = ''       # checkpoint to initialize with
    options['init_module'] = 'all'  # all/proposal/caption, which module to initialize


    ### MODEL CONFIG
    options['video_feat_dim'] = 500 # dim of image feature
    options['encoded_video_feat_dim'] = 512 # should be equal to rnn size
    options['word_embed_size'] = 512    # size of word embedding
    options['caption_seq_len'] = 30  # maximu length of a sentence
    options['num_rnn_layers'] = 2         # number of RNN layers
    options['rnn_size'] = 512        # hidden neuron size
    options['rnn_drop'] = 0.3        # rnn dropout
    options['num_anchors'] = 128     # number of anchors  
    options['no_context'] = False   # whether to use proposal context 
    options['context_gating'] = True  # whether to apply context gating
    options['max_proposal_len'] = 150    # max length of proposal allowed, used to construct a fixed length tensor for all proposals from one video
    options['attention_hidden_size'] = 512  # size of hidden neuron for the attention hidden layer

    
    ### OPTIMIZATION
    options['gpu_id'] = [0]    # GPU ids
    options['train_id'] = 1    # train id (useful when you have multiple runs)
    options['solver'] = 'adam' # 'adam','rmsprop','sgd_nestreov_momentum'
    options['momentum'] =  0.9     # only valid when solver is set to momentum optimizer
    options['batch_size'] = 1   # set to 1 to avoid different proposals problem, note that current implementation only supports batch_size=1
    options['eval_batch_size'] = 1
    options['loss_eval_num'] = 5000       # maximum evaluation batch number for loss
    options['metric_eval_num'] = 5000     # evaluation batch number for metric
    options['learning_rate'] = 1e-3       # initial learning rate
    options['lr_decay_factor'] = 0.1      # learning rate decay factor
    options['n_epoch_to_decay'] = list(range(20,60,20))[::-1]
    options['auto_lr_decay'] = True  # whether automatically decay learning rate based on val loss or evaluation score (only when evaluation_metric is True)
    options['n_eval_observe'] = 5   # if after 5 evaluations, the val loss is still not lower, go back to change learning rate 
    options['min_lr'] = 1e-5      # minimum learning rate allowed
    options['reg'] = 1e-6        # regularization strength
    options['init_scale'] = 0.08 # the init scale for uniform, here for initializing word embedding matrix
    options['max_epochs'] = 100  # maximum epochs
    options['init_epoch'] = 0    # initial epoch (useful when starting from last checkpoint)
    options['n_eval_per_epoch'] = 1 # number of evaluations per epoch
    options['eval_init'] = True # evaluate the initialized model
    options['shuffle'] = True

    options['clip_gradient_norm'] = 100.  # threshold to clip gradients: avoid gradient exploding problem; set to -1 to avoid gradient clipping
    options['log_input_min']  = 1e-20     # minimum input to the log() function
    options['weight_proposal'] = 0.1  # contribution weight of proposal module
    options['weight_caption'] = 1.0   # contribution weight of captioning module
    options['proposal_tiou_threshold'] = 0.5   # tiou threshold to positive samples, when changed, calculate class weights for positive/negative class again
    options['caption_tiou_threshold'] = 0.8    # tiou threshold to select high-iou proposals to feed in the captioning module
    options['predict_score_threshold'] = 0.5 # score threshold to select proposals at test time
    options['train_proposal'] = True   # whether to train variables of proposal module
    options['train_caption'] = True    # whether to train variables of captioning module
    options['evaluate_metric'] = True  # whether refer to evalutaion metric (CIDEr, METEOR, ...) for optimization


    ### INFERENCE
    options['tiou_measure'] = [0.3, 0.5, 0.7, 0.9]
    options['max_proposal_num'] = 100   # just for fast evaluation during training phase

    ### LOGGING
    options['ckpt_prefix'] = 'checkpoints/' + str(options['train_id']) + '/'  # where to save your checkpoints
    options['ckpt_sufix'] = ''
    options['status_file'] = options['ckpt_prefix'] + 'status.json'   # where to save your training status
    options['n_iters_display'] = 1   # frequency to display
    if not os.path.exists(options['ckpt_prefix']):
        os.mkdir(options['ckpt_prefix'])


    ### DEBUG
    options['print_debug'] = True   # 
    options['test_tensors'] = ['video_feat_fw', 'video_feat_bw', 'proposal_fw', 'proposal_bw', 'proposal_weight', 'proposal_score_fw', 'caption',
    'caption_mask', 'boolean_mask']  

    return options
    
