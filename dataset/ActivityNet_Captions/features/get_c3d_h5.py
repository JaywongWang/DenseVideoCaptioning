import os
import h5py
import numpy as np
import json

data = 'your_feature_numpys/'
feat_files = os.listdir(data)
feat_files = [file for file in feat_files if file.endswith('.npy')]

#video_info = json.load(open('activity_net.v1-3.min.json', 'r'))['database']

feat_dict = {}

print('Start ...')
count = 0
for file in feat_files:
    vid = file.split('.')[0]
    print(vid)
    filepath = data + file
    feat = np.load(filepath)
    '''
    duration = video_info[vid[2:]]['duration']
    n_feat = int((duration*24 - 8) // 8)
    feat = feat[:n_feat]
    np.save(filepath, feat)
    '''
    feat_dict[vid] = feat
    

    count += 1
    if count%1000 == 0:
        print('Processed %d files.'%count)

print('Processed %d files.'%count)

print('Writing file ...')

fid = h5py.File('your_feature_h5.hdf5', 'w')

for vid in feat_dict.keys():
    if vid in fid:
        print('WARNING: group name exists.')
        continue

    fid.create_group(vid).create_dataset('your_feature_name', data=feat_dict[vid])

print('Done.')


