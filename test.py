"""
Main function for testing
"""

import os
import numpy as np
import json
import h5py
from opt import *
from data_provider import *
from model import *

import tensorflow as tf
import sys

sys.path.insert(0, './densevid_eval-master')
sys.path.insert(0, './densevid_eval-master/coco-caption')
#from evaluator import *
from evaluator_old import *
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

reload(sys)
sys.setdefaultencoding('utf-8')

def getKey(item):
    return item['score']

"""
For joint ranking: combine proposal scores and caption confidence
"""
def getJointKey(item):
    #return 10.*item['proposal_score'] + item['sentence_confidence']
    return item['proposal_score'] + 5.*item['sentence_confidence']

"""
Generate batch data and corresponding mask data for the input
"""
def process_batch_data(batch_data, max_length):

    dim = batch_data[0].shape[1]

    out_batch_data = np.zeros(shape=(len(batch_data), max_length, dim), dtype='float32')
    out_batch_data_mask = np.zeros(shape=(len(batch_data), max_length), dtype='int32')

    for i, data in enumerate(batch_data):
        effective_len = min(max_length, data.shape[0])
        out_batch_data[i, :effective_len, :] = data[:effective_len]
        out_batch_data_mask[i, :effective_len] = 1

    out_batch_data = np.asarray(out_batch_data, dtype='float32')
    out_batch_data_mask = np.asarray(out_batch_data_mask, dtype='int32')

    return out_batch_data, out_batch_data_mask


def test(options):
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])[1:-1]
    sess = tf.InteractiveSession(config=sess_config)

    # build model
    print('Building model ...')
    model = CaptionModel(options)
    
    proposal_inputs, proposal_outputs = model.build_proposal_inference(reuse=False)
    caption_inputs, caption_outputs = model.build_caption_greedy_inference(reuse=False)

    # print variable names
    for v in tf.trainable_variables():
        print(v.name)
        print(v.get_shape())

    print('Restoring model from %s'%options['init_from'])
    saver = tf.train.Saver()
    saver.restore(sess, options['init_from'])

    word2ix = options['vocab']
    ix2word = {ix:word for word,ix in word2ix.items()}
    
    print('Start to predict ...')
    
    t0 = time.time()

    # output json data for evaluation
    out_data = {}
    out_data['version'] = 'VERSION 1.0'
    out_data['external_data'] = {'used':False, 'details': ''}
    out_data['results'] = {}
    results = {}

    count = 0

    split = 'val'

    test_ids = json.load(open('dataset/ActivityNet_Captions/%s_ids.json'%split, 'r'))
    #features = h5py.File('dataset/ActivityNet/features/c3d/sub_activitynet_v1-3_stride_64frame.c3d.hdf5', 'r')
    features = h5py.File('/data1/jwongwang/dataset/ActivityNet/features/c3d/activitynet_c3d_fc7_stride_64_frame.hdf5', 'r')
    stride = 4
    c3d_resolution = 16
    frame_rates = json.load(open('dataset/ActivityNet_Captions/video_fps.json'))
    anchor_data = open('dataset/ActivityNet_Captions/preprocess/anchors/anchors.txt', 'r').readlines()
    anchors = [float(line.strip()) for line in anchor_data]

    print('Will evaluate %d videos from %s set ...'%(len(test_ids), split))

    for vid in test_ids:
        print('\nProcessed %d-th video: %s'%(count, vid))

        # video feature sequence 
        video_feat_fw = features[vid]['c3d_fc7_features'].value
        video_feat_bw = np.flip(video_feat_fw, axis=0)
        video_feat_fw = np.expand_dims(video_feat_fw, axis=0)
        video_feat_bw = np.expand_dims(video_feat_bw, axis=0)


        proposal_score_fw, proposal_score_bw, rnn_outputs_fw, rnn_outputs_bw = sess.run([proposal_outputs['proposal_score_fw'], proposal_outputs['proposal_score_bw'], proposal_outputs['rnn_outputs_fw'], proposal_outputs['rnn_outputs_bw']], feed_dict={proposal_inputs['video_feat_fw']:video_feat_fw, proposal_inputs['video_feat_bw']:video_feat_bw})
        
        
        feat_len = video_feat_fw[0].shape[0]
        fps = frame_rates[vid]
        duration = feat_len*c3d_resolution*stride / fps


        '''calculate final score by summarizing forward score and backward score
        '''
        proposal_score = np.zeros((feat_len, options['num_anchors']))
        proposal_infos = []

        
        for i in range(feat_len):
            pre_start = -1.
            for j in range(options['num_anchors']):
                forward_score = proposal_score_fw[i,j]
                # calculate time stamp
                end = (float(i+1)/feat_len)*duration
                start = end-anchors[j]
                start = max(0., start)

                if start == pre_start:
                    continue

                # backward
                end_bw = duration - start
                i_bw = min(int(round((end_bw/duration)*feat_len)-1), feat_len-1)
                i_bw = max(i_bw, 0)
                backward_score = proposal_score_bw[i_bw,j]

                proposal_score[i,j] = forward_score*backward_score
                
                hidden_feat = np.concatenate([rnn_outputs_fw[i], rnn_outputs_bw[i_bw]], axis=-1)
                
                proposal_feats = video_feat_fw[0][feat_len-1-i_bw:i+1]
                proposal_infos.append({'timestamp':[start, end], 'score': proposal_score[i,j], 'event_hidden_feats': hidden_feat, 'proposal_feats': proposal_feats})
                            
                pre_start = start
        
        # add the largest proposal
        hidden_feat = np.concatenate([rnn_outputs_fw[feat_len-1], rnn_outputs_bw[feat_len-1]], axis=-1)
        
        proposal_feats = video_feat_fw[0]
        proposal_infos.append({'timestamp':[0., duration], 'score': 1., 'event_hidden_feats': hidden_feat, 'proposal_feats': proposal_feats})


        proposal_infos = sorted(proposal_infos, key=getKey, reverse=True)
        proposal_infos = proposal_infos[:options['max_proposal_num']]

        print('Number of proposals: %d'%len(proposal_infos))

        # 
        event_hidden_feats = [item['event_hidden_feats'] for item in proposal_infos]
        proposal_feats = [item['proposal_feats'] for item in proposal_infos]

        event_hidden_feats = np.array(event_hidden_feats, dtype='float32')
        proposal_feats, _ = process_batch_data(proposal_feats, options['max_proposal_len'])

        # word ids
        word_ids, word_confidences = sess.run([caption_outputs['word_ids'], caption_outputs['word_confidences']], feed_dict={caption_inputs['event_hidden_feats']: event_hidden_feats, caption_inputs['proposal_feats']: proposal_feats})
        
        sentences = [[ix2word[i] for i in ids] for ids in word_ids]
        sentences = [sentence[1:] for sentence in sentences]
        
        # remove <END> word
        out_sentences = []
        sentence_confidences = []
        for i, sentence in enumerate(sentences):
            end_id = options['caption_seq_len']
            if '<END>' in sentence:
                end_id = sentence.index('<END>')
                sentence = sentence[:end_id]
            sentence = ' '.join(sentence)
            sentence = sentence.replace('<UNK>', '') # remove unknown word token 
            out_sentences.append(sentence)
            
            if end_id <= 3:
                sentence_confidence = -1000.   # a very low score for very short sentence 
            else:
                sentence_confidence = float(np.mean(word_confidences[i, 1:end_id]))  # use np.mean instead of np.sum to avoid favoring short sentences
            sentence_confidences.append(sentence_confidence)

        
        print('Output sentences: ')
        for out_sentence in out_sentences:
            print(out_sentence.encode('utf-8'))


        result = [{'timestamp': proposal['timestamp'], 'proposal_score': float(proposal['score']), 'sentence': out_sentences[i], 'sentence_confidence': float(sentence_confidences[i])} for i, proposal in enumerate(proposal_infos)]

        # jointly ranking by proposal score and sentence confidence
        result = sorted(result, key=getJointKey, reverse=True)
                    
        results[vid] = result

        count += 1

    out_data['results'] = results
    
    resFile = 'results/%d/%s_%d_proposal_result.json'%(options['train_id'], split, options['max_proposal_num'])

    rootfolder = os.path.dirname(resFile)
    if not os.path.exists(rootfolder):
        print('Make directory %s ...'%rootfolder)
        os.makedirs(rootfolder)

    print('Saving result json file ...')
    with open(resFile, 'w') as fid:
        json.dump(out_data, fid)

    if split != 'val':
        print('Total running time: %f seconds.'%(time.time()-t0))
        return


    # Call evaluator
    
    evaluator = ANETcaptions(ground_truth_filenames=['densevid_eval-master/data/val_1.json', 'densevid_eval-master/data/val_2.json'],
                             prediction_filename=resFile,
                             tious=options['tiou_measure'],
                             #max_proposals=options['max_proposal_num'],
                             max_proposals=options['max_proposal_num'],
                             verbose=False)
    evaluator.evaluate()

    # Output the results
    for i, tiou in enumerate(options['tiou_measure']):
        print('-' * 80)
        print('tIoU: %.2f'%tiou)
        print('-' * 80)
        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            print('| %s: %2.4f'%(metric, 100*score))

    # Print the averages
    print('-' * 80)
    print('Average across all tIoUs')
    print('-' * 80)
    avg_scores = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        avg_score = 100 * sum(score) / float(len(score))
        avg_scores[metric] = avg_score
    
    # print output evaluation scores
    fid = open('%s_result_eval.txt'%split, 'w')
    for metric, score in avg_scores.items():
        print('%s: %.4f'%(metric, score))
        fid.write('%s: %.4f \n'%(metric, score))
    fid.close()
    
    print('Total running time: %f seconds.'%(time.time()-t0))


if __name__ == '__main__':

    options = default_options()
    test(options)
