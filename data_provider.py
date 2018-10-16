"""
Data provider for the built models
"""

import random
import numpy as np
import os
import h5py
from collections import OrderedDict
import json
from opt import *
import random
import math

np.set_printoptions(threshold=np.inf)

class DataProvision:
    def __init__(self, options):
        assert options['batch_size'] == 1
        self._options = options
        self._splits = {'train':'train', 'val':'val_1'}

        self._ids = {}  # video ids
        captions = {}
        self._sizes = {}
        print('Loading paragraph data ...')
        for split in self._splits:
            tmp_ids = open(os.path.join(self._options['caption_data_root'], split, 'ids.txt'), 'r').readlines()
            tmp_ids = [id.strip() for id in tmp_ids]
            self._ids[split] = tmp_ids

            self._sizes[split] = len(self._ids[split])
            
            tmp_captions = json.load(open(os.path.join(self._options['caption_data_root'], split, 'encoded_sentences.json'), 'r'))
            captions[split] = {tmp_ids[i]:tmp_captions[i] for i in range(len(tmp_ids))}

        # merge two caption dictionaries
        self._captions = {}
        for split in self._splits:
            self._captions = dict(self._captions.items() + captions[split].items())


        # feature dictionary
        print('Loading c3d features ...')
        features = h5py.File(self._options['feature_data_path'], 'r')
        self._feature_ids = features.keys()
        self._features = {video_id:np.asarray(features[video_id].values()[0]) for video_id in self._feature_ids}
        

        # load label weight data
        print('Loading label weight data ...')
        self._proposal_weight = json.load(open(os.path.join(self._options['caption_data_root'], 'anchors', 'weights.json')))
        if self._options['proposal_tiou_threshold'] != 0.5:
            raise ValueError('Might need to recalculate class weights to handle imbalance data')

        # get anchors
        print('Loading anchor data ...')
        anchor_path = os.path.join(self._options['caption_data_root'], 'anchors', 'anchors.txt')
        anchors = open(anchor_path).readlines()
        self._anchors = [float(line.strip()) for line in anchors]
        # time stamp data
        print('Loading localization data ...')
        self._localization = {}
        for split in self._splits:
            data = json.load(open(os.path.join(self._options['localization_data_path'], '%s.json'%self._splits[split])))
            self._localization[split] = data


        print('Done loading.')

            

    def get_size(self, split):
        return self._sizes[split]

    def get_ids(self, split):
        return self._ids[split]

    def get_anchors(self):
        return self._anchors

    def get_localization(self):
        return self._localization

    # process caption batch data into standard format
    def process_batch_paragraph(self, batch_paragraph):
        paragraph_length = []
        caption_length = []
        for captions in batch_paragraph:
            paragraph_length.append(len(captions))
            cap_len = []
            for caption in captions:
                cap_len.append(len(caption))
            
            caption_length.append(cap_len)

        caption_num = len(batch_paragraph[0])
        input_idx = np.zeros((len(batch_paragraph), caption_num, self._options['caption_seq_len']), dtype='int32')
        input_mask = np.zeros_like(input_idx)
        
        for i, captions in enumerate(batch_paragraph):
            for j in range(caption_num):
                caption = captions[j]
                effective_len = min(caption_length[i][j], self._options['caption_seq_len'])
                input_idx[i, j, 0:effective_len] = caption[:effective_len]
                input_mask[i, j, 0:effective_len-1] = 1

        return input_idx, input_mask

    # provide batch data
    def iterate_batch(self, split, batch_size):

        ids = list(self._ids[split])

        if split == 'train':
            print('Randomly shuffle training data ...')
            random.shuffle(ids)

        current = 0
        
        while True:

            batch_paragraph = []
            batch_feature_fw = []
            batch_feature_bw = []
            batch_proposal_fw = []
            batch_proposal_bw = []

            # train in pair, use one caption as common gt
            batch_proposal_caption_fw = [] # 0/1 to indicate whether to select the lstm state to feed into captioning module (based on tIoU)
            batch_proposal_caption_bw = [] # index to select corresponding backward feature

            i = 0 # batch_size = 1
            vid = ids[i+current]
            feature_fw = self._features[vid]
            feature_len = feature_fw.shape[0]

            if 'print_debug' in self._options and self._options['print_debug']:
                print('vid: %s'%vid)
                print('feature_len: %d'%feature_len)

            feature_bw = np.flip(feature_fw, axis=0)

            batch_feature_fw.append(feature_fw)
            batch_feature_bw.append(feature_bw)

            
            localization = self._localization[split][vid]
            timestamps = localization['timestamps']
            duration = localization['duration']

            # start and end time of the video stream
            start_time = 0.
            end_time = duration

            
            n_anchors = len(self._anchors)
            # ground truth proposal
            gt_proposal_fw = np.zeros(shape=(feature_len, n_anchors), dtype='int32')
            gt_proposal_bw = np.zeros(shape=(feature_len, n_anchors), dtype='int32')
            # ground truth proposal for feeding into captioning module
            gt_proposal_caption_fw = np.zeros(shape=(feature_len, ), dtype='int32')
            # corresponding backward index
            gt_proposal_caption_bw = np.zeros(shape=(feature_len, ), dtype='int32')
            # ground truth encoded caption in each time step
            gt_caption = [[0] for i in range(feature_len)]

            paragraph = self._captions[vid]
            
            assert self._options['caption_tiou_threshold'] >= self._options['proposal_tiou_threshold']
            
            # calculate ground truth labels
            for stamp_id, stamp in enumerate(timestamps):
                t1 = stamp[0]
                t2 = stamp[1]
                if t1 > t2:
                    temp = t1
                    t1 = t2
                    t2 = temp
                start = t1
                end = t2

                start_bw = duration - end
                end_bw = duration - start

                
                # if not end or if no overlap at all
                if end > end_time or start > end_time:
                    continue
                
                end_feat_id = max(int(round(end*feature_len/duration)-1), 0)
                start_feat_id = max(int(round(start*feature_len/duration) - 1), 0)

                mid_feature_id = int(round(((1.-self._options['proposal_tiou_threshold'])*end + self._options['proposal_tiou_threshold']*start) * feature_len / duration)) - 1
                mid_feature_id = max(0, mid_feature_id)

                for i in range(mid_feature_id, feature_len):
                    overlap = False
                    for anchor_id, anchor in enumerate(self._anchors):
                        end_pred = (float(i+1)/feature_len) * duration
                        start_pred = end_pred - anchor

                        intersection = max(0, min(end, end_pred) - max(start, start_pred))
                        union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
                        iou = float(intersection) / (union + 1e-8)


                        if iou > self._options['proposal_tiou_threshold']:
                            overlap = True
                            # the corresonding label of backward lstm
                            i_bw = feature_len - 1 - (start_feat_id+end_feat_id-i)
                            i_bw = max(min(i_bw, feature_len-1), 0)

                            
                            gt_proposal_fw[i, anchor_id] = 1
                            gt_proposal_bw[i_bw, anchor_id] = 1
                                
                        
                            if iou > self._options['caption_tiou_threshold']:
                                gt_proposal_caption_fw[i] = 1
                                gt_proposal_caption_bw[i] = i_bw
                                gt_caption[i] = paragraph[stamp_id]

                        elif overlap:
                            break
                                
            
            batch_proposal_fw.append(gt_proposal_fw)
            batch_proposal_bw.append(gt_proposal_bw)
            batch_proposal_caption_fw.append(gt_proposal_caption_fw)
            batch_proposal_caption_bw.append(gt_proposal_caption_bw)
            batch_paragraph.append(gt_caption)
            
            batch_caption, batch_caption_mask = self.process_batch_paragraph(batch_paragraph)

            batch_feature_fw = np.asarray(batch_feature_fw, dtype='float32')
            batch_feature_bw = np.asarray(batch_feature_bw, dtype='float32')
            batch_caption = np.asarray(batch_caption, dtype='int32')
            batch_caption_mask = np.asarray(batch_caption_mask, dtype='int32')

            batch_proposal_fw = np.asarray(batch_proposal_fw, dtype='int32')
            batch_proposal_bw = np.asarray(batch_proposal_bw, dtype='int32')
            batch_proposal_caption_fw = np.asarray(batch_proposal_caption_fw, dtype='int32')
            batch_proposal_caption_bw = np.asarray(batch_proposal_caption_bw, dtype='int32')
            

            # serve as a tuple
            batch_data = {'video_feat_fw': batch_feature_fw, 'video_feat_bw': batch_feature_bw, 'caption': batch_caption, 'caption_mask': batch_caption_mask, 'proposal_fw': batch_proposal_fw, 'proposal_bw': batch_proposal_bw, 'proposal_caption_fw': batch_proposal_caption_fw, 'proposal_caption_bw': batch_proposal_caption_bw, 'proposal_weight': np.array(self._proposal_weight)}

            
            yield batch_data

            current = current + batch_size
            
            if current + batch_size > self.get_size(split):
                current = 0
                # at the end of list, shuffle it
                if split == 'train':
                    print('Randomly shuffle training data ...')
                    random.shuffle(ids)
                    print('The new shuffled ids are:')
                    print('%s, %s, %s, ..., %s'%(ids[0], ids[1], ids[2], ids[-1]))
                    time.sleep(3)
                else:
                    break



