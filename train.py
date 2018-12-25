"""
Main function for training
"""

import os
import argparse
import numpy as np
from opt import *
from data_provider import *
from model import * 
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.insert(0, './densevid_eval-master')
sys.path.insert(0, './densevid_eval-master/coco-caption')
#from evaluator import *
from evaluator_old import *

def getKey(item):
    return item['score']


"""
Loss evaluation
"""
def evaluation(options, data_provision, sess, inputs, t_loss):
    val_loss_list = []
    val_proposal_loss_list =[]
    val_caption_loss_list = []
    val_count = min(data_provision.get_size('val'), options['loss_eval_num'])
    batch_size = options['batch_size']
    
    count = 0
    for batch_data in data_provision.iterate_batch('val', batch_size):
        print('Evaluating batch: #%d'%count)
        count += 1
        feed_dict = {inputs['rnn_drop']:0.}
        for key, value in batch_data.items():
            if key not in inputs:
                continue
            feed_dict[inputs[key]] = value

        loss, proposal_loss, caption_loss = sess.run(
                    t_loss,
                    feed_dict=feed_dict)
        val_loss_list.append(loss * batch_data['caption'].shape[0])
        val_proposal_loss_list.append(proposal_loss * batch_data['caption'].shape[0])
        val_caption_loss_list.append(caption_loss * batch_data['caption'].shape[0])

        if count >= val_count:
            break
    
    ave_val_loss = sum(val_loss_list) / float(val_count)
    ave_proposal_val_loss = sum(val_proposal_loss_list) / float(val_count)
    ave_caption_val_loss = sum(val_caption_loss_list) / float(val_count)
    return ave_val_loss, ave_proposal_val_loss, ave_caption_val_loss

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

def evaluation_metric_greedy(options, data_provision, sess, proposal_inputs, caption_inputs, proposal_outputs, caption_outputs):
    print('Evaluating caption scores ...')

    word2ix = options['vocab']
    ix2word = {ix:word for word,ix in word2ix.items()}
    
    # output json data for evaluation
    out_data = {}
    out_data['version'] = 'VERSION 1.0'
    out_data['external_data'] = {'used':False, 'details': ''}
    out_data['results'] = {}
    results = {}
    
    count = 0
    batch_size = options['eval_batch_size']    # default batch size to evaluate
    assert batch_size == 1
    
    eval_num = batch_size*options['metric_eval_num']
    print('Will evaluate %d samples'%eval_num)

    val_ids = data_provision.get_ids('val')[:eval_num]
    anchors = data_provision.get_anchors()
    localizaitons = data_provision.get_localization()

    for batch_data in data_provision.iterate_batch('val', batch_size):
        print('\nProcessed %d-th batch \n'%count)
        vid = val_ids[count]
        print('video id: %s'%vid)

        proposal_score_fw, proposal_score_bw, rnn_outputs_fw, rnn_outputs_bw = sess.run([proposal_outputs['proposal_score_fw'], proposal_outputs['proposal_score_bw'], proposal_outputs['rnn_outputs_fw'], proposal_outputs['rnn_outputs_bw']], feed_dict={proposal_inputs['video_feat_fw']:batch_data['video_feat_fw'], proposal_inputs['video_feat_bw']:batch_data['video_feat_bw']})
        
        feat_len = batch_data['video_feat_fw'][0].shape[0]
        duration = localizaitons['val'][vid]['duration']
        
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
                    
                
                proposal_feats = batch_data['video_feat_fw'][0][feat_len-1-i_bw:i+1]
                proposal_infos.append({'timestamp':[start, end], 'score': proposal_score[i,j], 'event_hidden_feats': hidden_feat, 'proposal_feats': proposal_feats})
                            
                pre_start = start
        
        # add the largest proposal
        hidden_feat = np.concatenate([rnn_outputs_fw[feat_len-1], rnn_outputs_bw[feat_len-1]], axis=-1)
            
        
        proposal_feats = batch_data['video_feat_fw'][0]
        proposal_infos.append({'timestamp':[0., duration], 'score': 1., 'event_hidden_feats': hidden_feat, 'proposal_feats': proposal_feats})
        

        proposal_infos = sorted(proposal_infos, key=getKey, reverse=True)
        proposal_infos = proposal_infos[:options['max_proposal_num']]

        print('Number of proposals: %d'%len(proposal_infos))

        # 
        event_hidden_feats = [item['event_hidden_feats'] for item in proposal_infos]
        proposal_feats = [item['proposal_feats'] for item in proposal_infos]

        
        event_hidden_feats = np.array(event_hidden_feats, dtype='float32')
        proposal_feats, _ = process_batch_data(proposal_feats, options['max_proposal_len'])

        # run session to get word ids
        word_ids = sess.run(caption_outputs['word_ids'], feed_dict={caption_inputs['event_hidden_feats']: event_hidden_feats, caption_inputs['proposal_feats']: proposal_feats})
        
        
        sentences = [[ix2word[i] for i in ids] for ids in word_ids]
        sentences = [sentence[1:] for sentence in sentences]
        
        # remove <END> word
        out_sentences = []
        for sentence in sentences:
            end_id = options['caption_seq_len']
            if '<END>' in sentence:
                end_id = sentence.index('<END>')
                sentence = sentence[:end_id]
            
            sentence = ' '.join(sentence)
            sentence = sentence.replace('<UNK>', '')
            out_sentences.append(sentence)

        
        print('Output sentences: ')
        for out_sentence in out_sentences:
            print(out_sentence)

        result = [{'timestamp': proposal['timestamp'], 'sentence': out_sentences[i]} for i, proposal in enumerate(proposal_infos)]
                    
        results[vid] = result

        count += 1

        if count >= eval_num:
            break

    out_data['results'] = results
    
    resFile = 'results/%d/temp_results.json'%options['train_id']
    root_folder = os.path.dirname(resFile)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    print('Saving result json file ...')
    with open(resFile, 'w') as fid:
        json.dump(out_data, fid)

    # Call evaluator
    
    evaluator = ANETcaptions(ground_truth_filenames=['densevid_eval-master/data/val_1.json', 'densevid_eval-master/data/val_2.json'],
                             prediction_filename=resFile,
                             tious=options['tiou_measure'],
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
    fid = open('results/%d/score_history.txt'%options['train_id'], 'a')
    for metric, score in avg_scores.items():
        print('%s: %.4f'%(metric, score))
        # also write to a temp file
        fid.write('%s: %.4f\n'%(metric, score))
    fid.write('\n')
    fid.close()

    combined_score = avg_scores['METEOR']
    
    return avg_scores, combined_score


def train(options):
    
    sess_config = tf.ConfigProto()
    #sess_config.gpu_options.allow_growth=True
    sess_config.gpu_options.allow_growth=False
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])[1:-1]
    sess = tf.InteractiveSession(config=sess_config)

    print('Load data ...')
    data_provision = DataProvision(options)

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    init_epoch = options['init_epoch']
    lr_init = options['learning_rate']
    status_file = options['status_file']
    lr = lr_init
    lr_decay_factor = options['lr_decay_factor']
    n_epoch_to_decay = options['n_epoch_to_decay'] # when to decay the lr
    next_epoch_to_decay = n_epoch_to_decay.pop()

    n_iters_per_epoch = data_provision.get_size('train') // batch_size
    eval_in_iters = n_iters_per_epoch // options['n_eval_per_epoch']

    #############################################
    # build model #

    print('Build model for training ...')
    model = CaptionModel(options)
    inputs, outputs = model.build_train()
    t_loss = outputs['loss']
    t_proposal_loss = outputs['proposal_loss']
    t_caption_loss = outputs['caption_loss']
    t_loss_list = [t_loss, t_proposal_loss, t_caption_loss]
    t_reg_loss = outputs['reg_loss']
    t_n_proposals = outputs['n_proposals']


    if options['evaluate_metric']:
        print('Build model for evaluating metric ...')
        proposal_inputs, proposal_outputs = model.build_proposal_inference(reuse=True)
        caption_inputs, caption_outputs = model.build_caption_greedy_inference(reuse=True)
        t_proposal_score_fw = proposal_outputs['proposal_score_fw']
        t_proposal_score_bw = proposal_outputs['proposal_score_bw']
        t_rnn_outputs_fw = proposal_outputs['rnn_outputs_fw']
        t_rnn_outputs_bw = proposal_outputs['rnn_outputs_bw']
        t_word_ids = caption_outputs['word_ids']
    #############################################
    
    t_summary = tf.summary.merge_all()
    t_lr = tf.placeholder(tf.float32)

    
    if options['solver'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=t_lr)
    elif options['solver'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=t_lr)
    elif options['solver'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=t_lr, momentum=options['momentum'])
    elif options['solver'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=t_lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=t_lr)
    
    # get trainable variable list
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if not options['train_proposal']:
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='caption_module')
    if not options['train_caption']:
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='proposal_module')
    if not options['train_proposal'] and not options['train_caption']:
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


    # gradient clipping option
    if options['clip_gradient_norm'] < 0:
        train_op = optimizer.minimize(t_loss + options['reg'] * t_reg_loss, var_list=trainable_vars)
    else:
        gvs = optimizer.compute_gradients(t_loss + options['reg'] * t_reg_loss, var_list=trainable_vars)
        clip_grad_var = [(tf.clip_by_norm(grad, options['clip_gradient_norm']), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var)

    # save summary data
    train_summary_writer = tf.summary.FileWriter(os.path.dirname(options['status_file']), sess.graph)

    # initialize all variables
    tf.global_variables_initializer().run()


    ## test model variable shape
    if 'print_debug' in options.keys() and options['print_debug']:
        print('*********** Variable Shape *************')
        for v in tf.trainable_variables():
            print('%s:'%v.name)
            print(v.get_shape())

        if 'test_tensors' in options:
            print('********** Tensor Shape ************')
            tf_graph = tf.get_default_graph()
            for t_name in options['test_tensors']:
                t = tf_graph.get_tensor_by_name('%s:0'%t_name)
                print('%s: '%t_name)
                print(t.get_shape())


    # for saving and restoring checkpoints during training
    saver = tf.train.Saver(max_to_keep=200, write_version=1)

    # initialize model from a given checkpoint path
    if options['init_from']:
        print('Init model from %s'%options['init_from'])
        pre_status = json.load(open(os.path.join(os.path.dirname(options['init_from']), 'status.json')))
        pre_options = pre_status['options']
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        if options['init_module'] == 'proposal':
            print('Restoring parameters only for proposal module')
            restore_vars = [v for v in restore_vars if v.name.startswith('proposal_module/')]
        elif options['init_module'] == 'caption':
            print('Restoring parameters only for caption module')
            restore_vars = [v for v in restore_vars if v.name.startswith('caption_module/')]
        elif options['init_module'] == 'all':
            pass
            
        # for restoring from another graph (contain different structure) at the beginning
        saver_part = tf.train.Saver(var_list=restore_vars)
        saver_part.restore(sess, options['init_from'])

    # save loss/evaluation history
    json_worker_status = OrderedDict()
    json_worker_status['options'] = options
    json_worker_status['history'] = []
    json_worker_status['eval_results'] = []
    json.dump(json_worker_status, open(options['status_file'], 'w'))


    if options['eval_init']:
        print('Evaluating the initialized model ...')
        val_loss, val_proposal_loss, val_caption_loss = evaluation(options, data_provision, sess, inputs, t_loss_list)
        print('loss: %.4f, proposal_loss: %.4f, caption_loss: %.4f'%(val_loss, val_proposal_loss, val_caption_loss))


        combined_score = -1   # denote not evaluated
        all_scores = -1
        if options['evaluate_metric']:
            all_scores, combined_score = evaluation_metric_greedy(options, data_provision, sess, proposal_inputs, caption_inputs, proposal_outputs, caption_outputs)
            
            print('combined score: %.3f'%(combined_score,))


    t0 = time.time()
    eval_id = 0
    train_batch_generator = data_provision.iterate_batch('train', batch_size)
    checkpoint_filenames = []
    
    for epoch in range(init_epoch, max_epochs):
        
        # manually set when to decay learning rate
        if not options['auto_lr_decay']:
            if epoch == next_epoch_to_decay:
                if len(n_epoch_to_decay) == 0:
                    next_epoch_to_decay = -1
                else:
                    next_epoch_to_decay = n_epoch_to_decay.pop()

                print('Decaying learning rate ...')
                lr *= lr_decay_factor
        
        print('epoch: %d/%d, lr: %.1E (%.1E)'%(epoch, max_epochs, lr, lr_init))
        for iter in range(n_iters_per_epoch):
            batch_data = next(train_batch_generator)
            feed_dict = {
                t_lr: lr,
                inputs['rnn_drop']: options['rnn_drop']
            }
            for key, value in batch_data.items():
                if key not in inputs:
                    continue
                feed_dict[inputs[key]] = value

            _, summary, loss, proposal_loss, caption_loss, reg_loss, n_proposals = sess.run([train_op, t_summary, t_loss, t_proposal_loss, t_caption_loss, t_reg_loss, t_n_proposals], feed_dict=feed_dict)

            if 'print_debug' in options and options['print_debug']:
                print('n_proposals: %d'%n_proposals)
            
            if iter == 0 and epoch == init_epoch:
                smooth_loss = loss
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * loss
            
            if iter % options['n_iters_display'] == 0:
                print('iter: %d, epoch: %d/%d, \nlr: %.1E, loss: %.4f, proposal_loss: %.4f, caption_loss: %.4f'%(iter, epoch, max_epochs, lr, loss, proposal_loss, caption_loss))
                train_summary_writer.add_summary(summary, iter + epoch * n_iters_per_epoch)
                jstatus = OrderedDict()
                jstatus['epoch'] = (epoch, max_epochs)
                jstatus['iter'] = (iter, n_iters_per_epoch)
                jstatus['loss'] = (float(loss), float(smooth_loss), float(reg_loss))
                json_worker_status['history'].append(jstatus)


            # every 30 secs write once
            if (time.time() - t0) / 60.0 > 0.5:
                t0 = time.time()
                json.dump(json_worker_status, open(status_file, 'w'))
            
            if (iter + 1) % eval_in_iters == 0:
                
                print('Evaluating model ...')
                val_loss, val_proposal_loss, val_caption_loss = evaluation(options, data_provision, sess, inputs, t_loss_list)
                print('loss: %.4f, proposal_loss: %.4f, caption_loss: %.4f'%(val_loss, val_proposal_loss, val_caption_loss))

                combined_score = -1   # denote not evaluated
                all_scores = -1
                if options['evaluate_metric']:
                    all_scores, combined_score = evaluation_metric_greedy(options, data_provision, sess, proposal_inputs, caption_inputs, proposal_outputs, caption_outputs)
                    
                    print('combined score: %.3f'%(combined_score,))

                jeval_results = OrderedDict()
                jeval_results['loss'] = (val_loss, smooth_loss)
                jeval_results['score'] = combined_score
                jeval_results['scores'] = all_scores
                jeval_results['lr'] = lr
                json_worker_status['eval_results'].append(jeval_results)
                json.dump(json_worker_status, open(status_file, 'w'))

                checkpoint_path = '%sepoch%02d_%.2f_%02d_lr%f%s.ckpt' % (options['ckpt_prefix'], epoch, val_loss, eval_id, lr, options['ckpt_sufix'])
                if options['evaluate_metric']:
                    checkpoint_path = '%sepoch%02d_%.2f_%02d_lr%f%s.ckpt' % (options['ckpt_prefix'], epoch, combined_score, eval_id, lr, options['ckpt_sufix'])

                saver.save(sess, checkpoint_path)
                checkpoint_filenames.append(checkpoint_path)
                
                eval_id = eval_id + 1

                # automatically lower learning rate
                if options['auto_lr_decay']:
                    # review val loss history or score history
                    eval_results = json_worker_status['eval_results']
                    view_end_eval_id = eval_id
                    view_start_eval_id = view_end_eval_id - options['n_eval_observe']
                    view_start_epoch_id = (view_end_eval_id + init_epoch*options['n_eval_per_epoch'] - options['n_eval_observe']) // options['n_eval_per_epoch']
                    

                    review_results = [result['loss'][0] for result in eval_results[view_start_eval_id:view_end_eval_id]]
                    if options['evaluate_metric']:
                        review_results = [result['score'] for result in eval_results[view_start_eval_id:view_end_eval_id]] 
                    
                    if view_start_eval_id >= 0:
                        if options['evaluate_metric'] and review_results.index(max(review_results)) == 0:
                            # go back to the state of view_start_eval_id, and lower learning rate  
                            print('Init model from %s ...'%checkpoint_filenames[view_start_eval_id])
                            saver.restore(sess, checkpoint_filenames[view_start_eval_id])
                            print('Decaying learning rate ...')
                            lr *= lr_decay_factor
                            if lr < options['min_lr']:
                                print('Reach minimum learning rate. Done training.')
                                return
                        elif not options['evaluate_metric'] and review_results.index(min(review_results)) == 0:
                            # go back to the state of view_start_eval_id, and lower learning rate
                            print('Init model from %s ...'%checkpoint_filenames[view_start_eval_id])
                            saver.restore(sess, checkpoint_filenames[view_start_eval_id])
                            print('Decaying learning rate ...')
                            lr *= lr_decay_factor
                            if lr < options['min_lr']:
                                print('Reach minimum learning rate. Done training.')
                                return
                
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = default_options()
    for key, value in options.items():
        parser.add_argument('--%s'%key, dest=key, type=type(value), default=None)
    args = parser.parse_args()
    args = vars(args)
    for key, value in args.items():
        if value:
            options[key] = value
            if key == 'ckpt_prefix':
                if not options['ckpt_prefix'].endswith('/'):
                    options['ckpt_prefix'] = options['ckpt_prefix'] + '/'
                options['status_file'] = options['ckpt_prefix'] + 'status.json'

    work_dir = os.path.dirname(options['status_file'])
    if os.path.exists(work_dir) :
        print('work_dir %s exists! Pls check it.'%work_dir)
    else:
        os.makedirs(work_dir)
    train(options)

