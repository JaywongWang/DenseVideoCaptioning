## since a video usually contains long sequence (in ActivityNet, 180s in average)
## I use local mean pooling to shorten the feature sequence
## 

import os
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_path', type=str, default='sub_activitynet_v1-3.c3d.hdf5', help='feature data path')
parser.add_argument('--short_ratio', type=int, default=8, help='short ratio')
parser.add_argument('--output_feature_path', type=str, default='activitynet_c3d_fc7_stride_64_frame.hdf5', help='output feature data path')
args = parser.parse_args()
feature_path = args.feature_path
short_ratio = args.short_ratio
output_feature_path = args.output_feature_path

features = h5py.File(feature_path, 'r')
out_features = {}
feat_dim = 500

for vid in features.keys():
    feat = np.asarray(features[vid].values()[0])
    seq_len = feat.shape[0]
    
    ll = seq_len%short_ratio
    padding = False
    
    if ll != 0:
        padding = True
        lt = int(seq_len/short_ratio)*short_ratio
        pad_feat = np.mean(feat[lt:lt+ll, :], axis=0)
        pad_feats = np.tile(pad_feat, (short_ratio-ll, 1))
        #pad_frames = np.zeros((short_ratio-ll, feat_dim))

    if padding:
        feat = np.concatenate((feat, pad_feats), axis=0)

    new_len = feat.shape[0]/short_ratio
    short_feat = []
    
    for i in range(new_len):
        short_feat.append(np.mean(feat[short_ratio*i:short_ratio*(i+1), :], axis=0))
    '''
    for i in range(new_len):
        short_feat.append(feat[short_ratio*(i+1)-1])
    '''

    short_feat = np.asarray(short_feat, dtype='float32')
    out_features[vid] = short_feat

print('Writing to hdf5 file ...')
fid = h5py.File(output_feature_path, 'w')
for vid in out_features.keys():
    if vid in fid:
        print('WARNING: group name exists.')
        continue
    fid.create_group(vid).create_dataset('c3d_fc7_features', data=out_features[vid])

print('Done.')

        
