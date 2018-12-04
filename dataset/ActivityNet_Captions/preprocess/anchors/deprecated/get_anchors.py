'''
This script is to get anchors and pos/neg weights 
'''
import os
import h5py
import json
import math
import numpy as np
import h5py
import random
import time
import threading
from sklearn.cluster import KMeans

sample_ratio = 1.0
c3d_resolution = 16
stride = 4
sample_num = 1
n_anchors = 128
tiou_threshold = 0.5
num_scale = 1


feature_path = '../../features/sub_activitynet_v1-3_stride_64frame.c3d.hdf5'
features = h5py.File(feature_path)

splits = {'train':'train', 'val':'val_1', 'test':'val_2'}

out_proposal_source = '../'
out_anchor_file = 'anchors.txt'


train_data = json.load(open('../../%s.json'%'train'))

video_ids = open(os.path.join(out_proposal_source, 'train', 'ids.txt')).readlines()
video_ids = [video_id.strip() for video_id in video_ids]


feature_lengths = dict()
proposal_lengths = []
for video_id in video_ids:
    data = train_data[video_id]
    timestamps = data['timestamps']
    duration = data['duration']
    feature = features[video_id].values()[0]
    feature_len = feature.shape[0]
    feature_lengths[video_id] = feature_len

    for stamp in timestamps:
        t1 = stamp[0]
        t2 = stamp[1]
        if t1 > t2:
            temp = t1
            t1 = t2
            t2 = temp
        
        clip_len = t2-t1
        proposal_lengths.append(clip_len)
        
    
proposal_lengths = np.array(proposal_lengths).reshape(len(proposal_lengths), 1)
print('Clustering all proposals ...')
kmeans = KMeans(n_clusters=n_anchors, random_state=0).fit(proposal_lengths)
anchors = kmeans.cluster_centers_
anchors = np.array(anchors.reshape(anchors.shape[0],), dtype=np.float32)
anchors = list(anchors)

# remove duplicate
anchors = sorted(list(set(anchors)))

print('Number of anchors: %d'%len(anchors))

# avoid inconsistency
n_anchors = len(anchors)


print('Writing anchors ...')
with open(out_anchor_file, 'w') as fid:
    fid.writelines([str(anchor)+'\n' for anchor in anchors])

# a thread to count the sampled video stream proposal
class get_proposal_label_thread (threading.Thread):
    def __init__(self, threadID, name, feature_lengths, train_data, video_ids, n_anchors, anchors, count_anchors):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.feature_lengths = feature_lengths
        self.train_data = train_data
        self.video_ids = video_ids
        self.n_anchors = n_anchors
        self.anchors = anchors
        self.count_anchors = count_anchors
        
    def run(self):
        print('%s start ...'%self.name)

        for index, vid in enumerate(self.video_ids):
            print('Processing video id: %s'%vid)
            data = self.train_data[vid]
            timestamps = data['timestamps']
            duration = data['duration']
            
            feature_len = self.feature_lengths[vid]
            print('feat len: %d'%feature_len)
            stream_len = int(math.ceil(sample_ratio*feature_len))
            start_feature_id = random.randint(0, feature_len-stream_len)
            end_feature_id = start_feature_id + stream_len

            
            start_time = (float(start_feature_id) / feature_len) * duration
            end_time = (float(end_feature_id) / feature_len) * duration

            print('sample stream: (%f, %f)'%(start_time, end_time))

            
            for stamp_id, stamp in enumerate(timestamps):
                t1 = stamp[0]
                t2 = stamp[1]
                if t1 > t2:
                    temp = t1
                    t1 = t2
                    t2 = temp
                start = t1
                end = t2

                
                start = max(start, start_time)

                # if not end or if no overlap at all
                if end > end_time or start > end_time:
                    continue

                mid_feature_id = int(((1.-tiou_threshold)*end + tiou_threshold*start) * feature_len / duration)
                for i in range(mid_feature_id, stream_len):
                    overlap = False
                    for anchor_id, anchor in enumerate(self.anchors):
                        # from feature id to time stamp
                        end_pred = (float(i+1)/feature_len) * duration
                        start_pred = end_pred - anchor

                        # 
                        if end_pred < end or i - int(end*feature_len/duration) > 5:
                            continue

                        intersection = max(0, min(end, end_pred) - max(start, start_pred))
                        union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
                        iou = float(intersection) / (union + 1e-8)


                        if iou > tiou_threshold:
                            self.count_anchors[self.threadID][anchor_id] += 1
                            overlap = True
                            
                        elif overlap:
                            break



def get_proposal_label(num_thread, feature_lengths, train_data, video_ids, n_anchors, achors, count_anchors):
    threads = []
    for i in range(num_thread):
        
        this_thread = get_proposal_label_thread(i, 'Thread-%d'%i, feature_lengths, train_data, video_ids, n_anchors, anchors, count_anchors)
        this_thread.start()
        threads.append(this_thread)
        
    for thread in threads:
        thread.join()
        
    print('Exiting main thread.')



count_anchors = [[0 for _ in range(n_anchors)] for _ in range(sample_num)]
sum_video_length = sample_ratio*sum(feature_lengths.values())
weights = [[0., 0.] for _ in range(n_anchors)]
out_weight_path = 'weights.json'

# samplg to get anchor weights
print('Get anchor weights by sampling ...')
get_proposal_label(sample_num, feature_lengths, train_data, video_ids, n_anchors, anchors, count_anchors)

count_anchors = np.mean(np.array(count_anchors), axis=0)
for i in range(n_anchors):
    # weight for negative label
    weights[i][1] = count_anchors[i] / float(sum_video_length)
    # weight for positive label
    weights[i][0] = 1. - weights[i][1]


print('The weights are:')
print(weights)
print('Writing ...')
with open(out_weight_path, 'w') as fid:
    json.dump(weights, fid)

with open('weights.txt', 'w') as fid:
    for w in weights:
        fid.write('%.4f\t%.4f\n'%(w[0], w[1]))


