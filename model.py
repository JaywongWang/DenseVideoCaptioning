"""
Model definition
Implementation of dense captioning model in the paper "Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning" by Jingwen Wang et al. in CVPR, 2018.
The code looks complicated since we need to handle some "dynamic" part of the graph
"""

import tensorflow as tf

class CaptionModel(object):

    def __init__(self, options):
        self.options = options
        self.initializer = tf.random_uniform_initializer(
            minval = - self.options['init_scale'],
            maxval = self.options['init_scale'])

        tf.set_random_seed(options['random_seed'])

    """ 
    build video feature embedding
    """
    def build_video_feat_embedding(self, video_feat, reuse=False):
        with tf.variable_scope('video_feat_embed', reuse=reuse) as scope:
            video_feat_embed = tf.contrib.layers.fully_connected(
                inputs=video_feat,
                num_outputs=self.options['word_embed_size'],
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
        return video_feat_embed

    """
    build word embedding for each word in a caption
    """
    def build_caption_embedding(self, caption, reuse=False):
        with tf.variable_scope('word_embed', reuse=reuse):
            embed_map = tf.get_variable(
                name='map',
                shape=(self.options['vocab_size'], self.options['word_embed_size']),
                initializer=self.initializer
            )
            caption_embed = tf.nn.embedding_lookup(embed_map, caption)
        return caption_embed

    
    """
    Build graph for proposal generation (inference)
    """
    def build_proposal_inference(self, reuse=False):
        inputs = {}
        outputs = {}

        # this line of code is just a message to inform that batch size should be set to 1 only
        batch_size = 1

        #******************** Define Proposal Module ******************#

        ## dim1: batch, dim2: video sequence length, dim3: video feature dimension
        ## video feature sequence

        # forward
        video_feat_fw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_fw')
        inputs['video_feat_fw'] = video_feat_fw

        # backward
        video_feat_bw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_bw')
        inputs['video_feat_bw'] = video_feat_bw
        
        rnn_cell_video_fw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video_bw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )

        with tf.variable_scope('proposal_module', reuse=reuse) as proposal_scope:

            '''video feature sequence encoding: forward pass
            '''
            with tf.variable_scope('video_encoder_fw', reuse=reuse) as scope:
                sequence_length = tf.expand_dims(tf.shape(video_feat_fw)[1], axis=0)
                initial_state = rnn_cell_video_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_fw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_fw, 
                    inputs=video_feat_fw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                
                
            rnn_outputs_fw_reshape = tf.reshape(rnn_outputs_fw, [-1, self.options['rnn_size']], name='rnn_outputs_fw_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_fw', reuse=reuse) as scope:
                logit_output_fw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_fw_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )

            '''video feature sequence encoding: backward pass
            '''
            with tf.variable_scope('video_encoder_bw', reuse=reuse) as scope:
                #sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                sequence_length = tf.expand_dims(tf.shape(video_feat_bw)[1], axis=0)
                initial_state = rnn_cell_video_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_bw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_bw, 
                    inputs=video_feat_bw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                
                
            rnn_outputs_bw_reshape = tf.reshape(rnn_outputs_bw, [-1, self.options['rnn_size']], name='rnn_outputs_bw_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_bw', reuse=reuse) as scope:
                logit_output_bw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_bw_reshape,
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )

        # score
        proposal_score_fw = tf.sigmoid(logit_output_fw, name='proposal_score_fw')
        proposal_score_bw = tf.sigmoid(logit_output_bw, name='proposal_score_bw')
        
        # outputs from proposal module
        outputs['proposal_score_fw'] = proposal_score_fw
        outputs['proposal_score_bw'] = proposal_score_bw
        outputs['rnn_outputs_fw'] = rnn_outputs_fw_reshape
        outputs['rnn_outputs_bw'] = rnn_outputs_bw_reshape


        return inputs, outputs

    """
    Build graph for caption generation (inference)
    Surprisingly, I found using beam search leads to worse meteor score on ActivityNet Captions dataset; similar observation has been found by other dense captioning papers
    I do not use beam search when generating captions
    """
    def build_caption_greedy_inference(self, reuse=False):
        inputs = {}
        outputs = {}

        # proposal feature sequences (the localized proposals/events can be of different length, I set a 'max_proposal_len' to make it easy for GPU processing)
        proposal_feats = tf.placeholder(tf.float32, [None, self.options['max_proposal_len'], self.options['video_feat_dim']])
        # combination of forward and backward hidden state, which encode event context information
        event_hidden_feats = tf.placeholder(tf.float32, [None, 2*self.options['rnn_size']])

        inputs['event_hidden_feats'] = event_hidden_feats
        inputs['proposal_feats'] = proposal_feats

        # batch size for inference, depends on how many proposals are generated for a video
        eval_batch_size = tf.shape(proposal_feats)[0]
        
        # intialize the rnn cell for captioning
        rnn_cell_caption = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )

        def get_rnn_cell():
            return tf.contrib.rnn.LSTMCell(num_units=self.options['rnn_size'], state_is_tuple=True, initializer=tf.orthogonal_initializer())

        # multi-layer LSTM
        multi_rnn_cell_caption = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])], state_is_tuple=True)

        # start word
        word_id = tf.fill([eval_batch_size], self.options['vocab']['<START>'])
        word_id = tf.to_int64(word_id)
        word_ids = tf.expand_dims(word_id, axis=-1)

        # probability (confidence) for the predicted word
        word_confidences = tf.expand_dims(tf.fill([eval_batch_size], 1.), axis=-1)

        # initial state of caption generation
        initial_state = multi_rnn_cell_caption.zero_state(batch_size=eval_batch_size, dtype=tf.float32)
        state = initial_state

        with tf.variable_scope('caption_module', reuse=reuse) as caption_scope:

            # initialize memory cell and hidden output, note that the returned state is a tuple containing all states for each cell in MultiRNNCell
            state = multi_rnn_cell_caption.zero_state(batch_size=eval_batch_size, dtype=tf.float32)

            proposal_feats_reshape = tf.reshape(proposal_feats, [-1, self.options['video_feat_dim']], name='video_feat_reshape')


            ## the caption data should be prepared in equal length, namely, with length of 'caption_seq_len'
            ## use caption mask data to mask out loss from sequence after end of token (<END>)
            # only the first loop create variable, the other loops reuse them, need to give variable scope name to each variable, otherwise tensorflow will create a new one
            for i in range(self.options['caption_seq_len']-1):

                if i > 0:
                    caption_scope.reuse_variables()

                # word embedding
                word_embed = self.build_caption_embedding(word_id)

                # get attention, receive both hidden state information (previous generated words) and video feature
                # state[:, 1] return all hidden states for all cells in MultiRNNCell
                h_state = tf.concat([s[1] for s in state], axis=-1)
                h_state_tile = tf.tile(h_state, [1, self.options['max_proposal_len']])
                h_state_reshape = tf.reshape(h_state_tile, [-1, self.options['num_rnn_layers']*self.options['rnn_size']])
                
                # repeat to match each feature vector in the localized proposal
                event_hidden_feats_tile = tf.tile(event_hidden_feats, [1, self.options['max_proposal_len']])
                event_hidden_feats_reshape = tf.reshape(event_hidden_feats_tile, [-1, 2*self.options['rnn_size']])

                feat_state_concat = tf.concat([proposal_feats_reshape, h_state_reshape, event_hidden_feats_reshape], axis=-1, name='feat_state_concat')


                # use a two-layer network to model temporal soft attention over proposal feature sequence when predicting next word (dynamic)
                with tf.variable_scope('attention', reuse=reuse) as attention_scope:
                    attention_layer1 = tf.contrib.layers.fully_connected(
                        inputs = feat_state_concat,
                        num_outputs = self.options['attention_hidden_size'],
                        activation_fn = tf.nn.tanh,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    ) 
                    attention_layer2 = tf.contrib.layers.fully_connected(
                        inputs = attention_layer1,
                        num_outputs = 1,
                        activation_fn = None,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    )

                # reshape to match
                attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_proposal_len']], name='attention_reshape')
                attention_score = tf.nn.softmax(attention_reshape, dim=-1, name='attention_score')
                attention = tf.reshape(attention_score, [-1, 1, self.options['max_proposal_len']], name='attention')

                # attended video feature
                attended_proposal_feat = tf.matmul(attention, proposal_feats, name='attended_proposal_feat')
                attended_proposal_feat_reshape = tf.reshape(attended_proposal_feat, [-1, self.options['video_feat_dim']], name='attended_proposal_feat_reshape')

                # whether to use proposal contexts to help generate the corresponding caption
                if self.options['no_context']:
                    proposal_feats_full = attended_proposal_feat_reshape
                else:
                    # whether to use gating function to combine the proposal contexts
                    if self.options['context_gating']:
                        # model a gate to weight each element of context and feature
                        attended_proposal_feat_reshape = tf.nn.tanh(attended_proposal_feat_reshape)
                        with tf.variable_scope('context_gating', reuse=reuse):
                            context_feats_transform = tf.contrib.layers.fully_connected(
                                inputs=event_hidden_feats,
                                num_outputs=self.options['video_feat_dim'],
                                activation_fn=tf.nn.tanh,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            
                            
                            gate = tf.contrib.layers.fully_connected(
                                inputs=tf.concat([word_embed, h_state, context_feats_transform, attended_proposal_feat_reshape], axis=-1),
                                num_outputs=self.options['video_feat_dim'],
                                activation_fn=tf.nn.sigmoid,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )

                            gated_context_feats = tf.multiply(context_feats_transform, gate)
                            gated_proposal_feats = tf.multiply(attended_proposal_feat_reshape, 1.-gate)
                            proposal_feats_full = tf.concat([gated_context_feats, gated_proposal_feats], axis=-1)
                            
                    else:
                        proposal_feats_full = tf.concat([event_hidden_feats, attended_proposal_feat_reshape], axis=-1)

                # proposal feature embedded into word space
                proposal_feat_embed = self.build_video_feat_embedding(proposal_feats_full)

                # get next state
                caption_output, state = multi_rnn_cell_caption(tf.concat([proposal_feat_embed, word_embed], axis=-1), state)

                # predict next word
                with tf.variable_scope('logits', reuse=reuse) as logits_scope:
                    logits = tf.contrib.layers.fully_connected(
                        inputs=caption_output,
                        num_outputs=self.options['vocab_size'],
                        activation_fn=None
                    )

                softmax = tf.nn.softmax(logits, name='softmax')
                word_id = tf.argmax(softmax, axis=-1)
                word_confidence = tf.reduce_max(softmax, axis=-1)
                word_ids = tf.concat([word_ids, tf.expand_dims(word_id, axis=-1)], axis=-1)
                word_confidences = tf.concat([word_confidences, tf.expand_dims(word_confidence, axis=-1)], axis=-1)

        sentence_confidences = tf.reduce_sum(tf.log(tf.clip_by_value(word_confidences, 1e-20, 1.)), axis=-1)

        outputs['word_ids'] = word_ids
        outputs['sentence_confidences'] = sentence_confidences

        return inputs, outputs


    """
    Build graph for training
    """
    def build_train(self):

        # this line of code is just a message to inform that batch size should be set to 1 only
        batch_size = 1

        inputs = {}
        outputs = {}

        #******************** Define Proposal Module ******************#

        ## dim1: batch, dim2: video sequence length, dim3: video feature dimension
        ## video feature sequence
        
        # forward video feature sequence
        video_feat_fw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_fw')
        inputs['video_feat_fw'] = video_feat_fw

        # backward video feature sequence
        video_feat_bw = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat_bw')
        inputs['video_feat_bw'] = video_feat_bw

        ## proposal data, densely annotated, in forward direction
        proposal_fw = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal_fw')
        inputs['proposal_fw'] = proposal_fw

        ## proposal data, densely annotated, in backward direction
        proposal_bw = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal_bw')
        inputs['proposal_bw'] = proposal_bw

        ## proposal to feed into captioning module, i choose high tiou proposals for training captioning module, forward pass
        proposal_caption_fw = tf.placeholder(tf.int32, [None, None], name='proposal_caption_fw')
        inputs['proposal_caption_fw'] = proposal_caption_fw

        ## proposal to feed into captioning module, i choose high tiou proposals for training captioning module, backward pass
        proposal_caption_bw = tf.placeholder(tf.int32, [None, None], name='proposal_caption_bw')
        inputs['proposal_caption_bw'] = proposal_caption_bw

        ## weighting for positive/negative labels (solve imbalance data problem)
        proposal_weight = tf.placeholder(tf.float32, [self.options['num_anchors'], 2], name='proposal_weight')
        inputs['proposal_weight'] = proposal_weight
        
        
        rnn_cell_video_fw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video_bw = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )

        if self.options['rnn_drop'] > 0:
            print('using dropout in rnn!')
            
        rnn_drop = tf.placeholder(tf.float32)
        inputs['rnn_drop'] = rnn_drop
        
        rnn_cell_video_fw = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_video_fw,
            input_keep_prob=1.0 - rnn_drop,
            output_keep_prob=1.0 - rnn_drop 
        )
        rnn_cell_video_bw = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_video_bw,
            input_keep_prob=1.0 - rnn_drop,
            output_keep_prob=1.0 - rnn_drop 
        )
        
        
        with tf.variable_scope('proposal_module') as proposal_scope:

            '''video feature sequence encoding: forward pass
            '''
            with tf.variable_scope('video_encoder_fw') as scope:
                #sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                sequence_length = tf.expand_dims(tf.shape(video_feat_fw)[1], axis=0)
                initial_state = rnn_cell_video_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_fw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_fw, 
                    inputs=video_feat_fw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
            
            rnn_outputs_fw_reshape = tf.reshape(rnn_outputs_fw, [-1, self.options['rnn_size']], name='rnn_outputs_fw_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_fw') as scope:
                logit_output_fw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_fw_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )

            '''video feature sequence encoding: backward pass
            '''
            with tf.variable_scope('video_encoder_bw') as scope:
                #sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                sequence_length = tf.expand_dims(tf.shape(video_feat_bw)[1], axis=0)
                initial_state = rnn_cell_video_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
                
                rnn_outputs_bw, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell_video_bw, 
                    inputs=video_feat_bw, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
            
            rnn_outputs_bw_reshape = tf.reshape(rnn_outputs_bw, [-1, self.options['rnn_size']], name='rnn_outputs_bw_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal_bw') as scope:
                logit_output_bw = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_bw_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )

        # calculate multi-label loss: use weighted binary cross entropy objective
        proposal_fw_reshape = tf.reshape(proposal_fw, [-1, self.options['num_anchors']], name='proposal_fw_reshape')
        proposal_fw_float = tf.to_float(proposal_fw_reshape)
        proposal_bw_reshape = tf.reshape(proposal_bw, [-1, self.options['num_anchors']], name='proposal_bw_reshape')
        proposal_bw_float = tf.to_float(proposal_bw_reshape)

        # weighting positive samples
        weight0 = tf.reshape(proposal_weight[:, 0], [-1, self.options['num_anchors']])
        # weighting negative samples
        weight1 = tf.reshape(proposal_weight[:, 1], [-1, self.options['num_anchors']])

        # tile weight batch_size times
        weight0 = tf.tile(weight0, [tf.shape(logit_output_fw)[0], 1])
        weight1 = tf.tile(weight1, [tf.shape(logit_output_fw)[0], 1])

        # get weighted sigmoid xentropy loss
        loss_term_fw = tf.nn.weighted_cross_entropy_with_logits(targets=proposal_fw_float, logits=logit_output_fw, pos_weight=weight0)
        loss_term_bw = tf.nn.weighted_cross_entropy_with_logits(targets=proposal_bw_float, logits=logit_output_bw, pos_weight=weight0)

        loss_term_fw_sum = tf.reduce_sum(loss_term_fw, axis=-1, name='loss_term_fw_sum')
        loss_term_bw_sum = tf.reduce_sum(loss_term_bw, axis=-1, name='loss_term_bw_sum')

        
        proposal_fw_loss = tf.reduce_sum(loss_term_fw_sum) / (float(self.options['num_anchors'])*tf.to_float(tf.shape(video_feat_fw)[1]))
        proposal_bw_loss = tf.reduce_sum(loss_term_bw_sum) / (float(self.options['num_anchors'])*tf.to_float(tf.shape(video_feat_bw)[1]))
        proposal_loss = (proposal_fw_loss + proposal_bw_loss) / 2.

        # summary data, for visualization using Tensorboard
        tf.summary.scalar('proposal_fw_loss', proposal_fw_loss)
        tf.summary.scalar('proposal_bw_loss', proposal_bw_loss)
        tf.summary.scalar('proposal_loss', proposal_loss)

        # outputs from proposal module
        outputs['proposal_fw_loss'] = proposal_fw_loss
        outputs['proposal_bw_loss'] = proposal_bw_loss
        outputs['proposal_loss'] = proposal_loss


        #*************** Define Captioning Module *****************#

        ## caption data: densely annotate sentences for each time step of a video, use mask data to mask out time steps when no caption should be output
        caption = tf.placeholder(tf.int32, [None, None, self.options['caption_seq_len']], name='caption')
        caption_mask = tf.placeholder(tf.int32, [None, None, self.options['caption_seq_len']], name='caption_mask')
        inputs['caption'] = caption
        inputs['caption_mask'] = caption_mask

        proposal_caption_fw_reshape = tf.reshape(proposal_caption_fw, [-1], name='proposal_caption_fw_reshape')
        proposal_caption_bw_reshape = tf.reshape(proposal_caption_bw, [-1], name='proposal_caption_bw_reshape')

        # use correct or 'nearly correct' proposal output as input to the captioning module
        boolean_mask = tf.greater(proposal_caption_fw_reshape, 0, name='boolean_mask')

        # guarantee that at least one pos has True value
        boolean_mask = tf.cond(tf.equal(tf.reduce_sum(tf.to_int32(boolean_mask)), 0), lambda: tf.concat([boolean_mask[:-1], tf.constant([True])], axis=-1), lambda: boolean_mask)

        # select input video state
        feat_len = tf.shape(video_feat_fw)[1]
        forward_indices = tf.boolean_mask(tf.range(feat_len), boolean_mask)
        event_feats_fw = tf.boolean_mask(rnn_outputs_fw_reshape, boolean_mask)
        backward_indices = tf.boolean_mask(proposal_caption_bw_reshape, boolean_mask)
        event_feats_bw = tf.gather_nd(rnn_outputs_bw_reshape, tf.expand_dims(backward_indices, axis=-1))
        
        start_ids = feat_len - 1 - backward_indices
        end_ids = forward_indices
        
        event_c3d_seq, _ = self.get_c3d_seq(video_feat_fw[0], start_ids, end_ids, self.options['max_proposal_len'])
        context_feats_fw = tf.gather_nd(rnn_outputs_fw_reshape, tf.expand_dims(start_ids, axis=-1))
        context_feats_bw = tf.gather_nd(rnn_outputs_bw_reshape, tf.expand_dims(feat_len-1-end_ids, axis=-1))

        # proposal feature sequences
        proposal_feats = event_c3d_seq

        # corresponding caption ground truth (batch size  = 1)
        caption_proposed = tf.boolean_mask(caption[0], boolean_mask, name='caption_proposed')
        caption_mask_proposed = tf.boolean_mask(caption_mask[0], boolean_mask, name='caption_mask_proposed')

        # the number of proposal-caption pairs for training
        n_proposals = tf.shape(caption_proposed)[0]

        rnn_cell_caption = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True, 
            initializer=tf.orthogonal_initializer()
        )
        
        rnn_cell_caption = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_caption,
            input_keep_prob=1.0 - rnn_drop,
            output_keep_prob=1.0 - rnn_drop 
        )

        def get_rnn_cell():
            return tf.contrib.rnn.LSTMCell(num_units=self.options['rnn_size'], state_is_tuple=True, initializer=tf.orthogonal_initializer())

        # multi-layer LSTM
        multi_rnn_cell_caption = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])], state_is_tuple=True)

        caption_loss = 0
        with tf.variable_scope('caption_module') as caption_scope:

            batch_size = n_proposals

            # initialize memory cell and hidden output, note that the returned state is a tuple containing all states for each cell in MultiRNNCell
            state = multi_rnn_cell_caption.zero_state(batch_size=batch_size, dtype=tf.float32)

            proposal_feats_reshape = tf.reshape(proposal_feats, [-1, self.options['video_feat_dim']], name='proposal_feats_reshape')

            event_hidden_feats = tf.concat([event_feats_fw, event_feats_bw], axis=-1)

            event_hidden_feats_tile = tf.tile(event_hidden_feats, [1, self.options['max_proposal_len']])
            event_hidden_feats_reshape = tf.reshape(event_hidden_feats_tile, [-1, 2*self.options['rnn_size']])


            ''' 
            The caption data should be prepared in equal length, namely, with length of 'caption_seq_len'
            ## use caption mask data to mask out loss from sequence after end of token (<END>)
            Only the first loop create variable, the other loops reuse them
            '''
            for i in range(self.options['caption_seq_len']-1):

                if i > 0:
                    caption_scope.reuse_variables()

                # word embedding
                word_embed = self.build_caption_embedding(caption_proposed[:, i])

                # calculate attention over proposal feature elements
                # state[:, 1] return all hidden states for all cells in MultiRNNCell
                h_state = tf.concat([s[1] for s in state], axis=-1)
                h_state_tile = tf.tile(h_state, [1, self.options['max_proposal_len']])
                h_state_reshape = tf.reshape(h_state_tile, [-1, self.options['num_rnn_layers']*self.options['rnn_size']])
                
                feat_state_concat = tf.concat([proposal_feats_reshape, h_state_reshape, event_hidden_feats_reshape], axis=-1, name='feat_state_concat')

                # use a two-layer network to model attention over video feature sequence when predicting next word (dynamic)
                with tf.variable_scope('attention') as attention_scope:
                    attention_layer1 = tf.contrib.layers.fully_connected(
                        inputs = feat_state_concat,
                        num_outputs = self.options['attention_hidden_size'],
                        activation_fn = tf.nn.tanh,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    )
                    attention_layer2 = tf.contrib.layers.fully_connected(
                        inputs = attention_layer1,
                        num_outputs = 1,
                        activation_fn = None,
                        weights_initializer = tf.contrib.layers.xavier_initializer()
                    )

                # reshape to match
                attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_proposal_len']], name='attention_reshape')
                attention_score = tf.nn.softmax(attention_reshape, dim=-1, name='attention_score')
                attention = tf.reshape(attention_score, [-1, 1, self.options['max_proposal_len']], name='attention')

                # attended video feature
                attended_proposal_feat = tf.matmul(attention, proposal_feats, name='attended_proposal_feat')
                attended_proposal_feat_reshape = tf.reshape(attended_proposal_feat, [-1, self.options['video_feat_dim']], name='attended_proposal_feat_reshape')

                
                if self.options['no_context']:
                    proposal_feats_full = attended_proposal_feat_reshape
                else:
                    if self.options['context_gating']:
                        # model a gate to weight each element of context and feature
                        attended_proposal_feat_reshape = tf.nn.tanh(attended_proposal_feat_reshape)
                        with tf.variable_scope('context_gating'):
                            context_feats_transform = tf.contrib.layers.fully_connected(
                                inputs=event_hidden_feats,
                                num_outputs=self.options['video_feat_dim'],
                                activation_fn=tf.nn.tanh,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            
                            # context gating
                            gate = tf.contrib.layers.fully_connected(
                                inputs=tf.concat([word_embed, h_state, context_feats_transform, attended_proposal_feat_reshape], axis=-1),
                                num_outputs=self.options['video_feat_dim'],
                                activation_fn=tf.nn.sigmoid,
                                weights_initializer=tf.contrib.layers.xavier_initializer()
                            )
                            gated_context_feats = tf.multiply(context_feats_transform, gate)
                            gated_proposal_feats = tf.multiply(attended_proposal_feat_reshape, 1.-gate)
                            proposal_feats_full = tf.concat([gated_context_feats, gated_proposal_feats], axis=-1)
                            
                    else:
                        proposal_feats_full = tf.concat([event_hidden_feats, attended_proposal_feat_reshape], axis=-1)


                # proposal feature embedded into word space
                proposal_feat_embed = self.build_video_feat_embedding(proposal_feats_full)

                # get next state
                caption_output, state = multi_rnn_cell_caption(tf.concat([proposal_feat_embed, word_embed], axis=-1), state)

                # predict next word
                with tf.variable_scope('logits') as logits_scope:
                    logits = tf.contrib.layers.fully_connected(
                        inputs=caption_output,
                        num_outputs=self.options['vocab_size'],
                        activation_fn=None
                    )

                labels = caption_proposed[:, i+1] # predict next word

                # loss term
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                output_mask = tf.to_float(caption_mask_proposed[:,i])
                loss = tf.reduce_sum(tf.multiply(loss, output_mask))
                
                caption_loss = caption_loss + loss

        # mean loss for each word
        caption_loss = caption_loss / (tf.to_float(batch_size)*tf.to_float(tf.reduce_sum(caption_mask_proposed)) + 1)

        tf.summary.scalar('caption_loss', caption_loss)
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith('caption_module/word_embed')])
        total_loss = self.options['weight_proposal']*proposal_loss + self.options['weight_caption']*caption_loss
        tf.summary.scalar('total_loss', total_loss)

        outputs['caption_loss'] = caption_loss
        outputs['loss'] = total_loss
        outputs['reg_loss'] = reg_loss
        outputs['n_proposals'] = n_proposals

        return inputs, outputs


    """get c3d proposal representation (feature sequence), given start end feature ids
    """
    def get_c3d_seq(self, video_feat_sequence, start_ids, end_ids, max_clip_len):
        
        ind = tf.constant(0)
        N = tf.shape(start_ids)[0]
        event_c3d_sequence = tf.fill([0, max_clip_len, self.options['video_feat_dim']], 0.)
        event_c3d_mask = tf.fill([0, max_clip_len], 0.)
        event_c3d_mask = tf.to_int32(event_c3d_mask)

        def condition(ind, event_c3d_sequence, event_c3d_mask):
            return tf.less(ind, N)

        def body(ind, event_c3d_sequence, event_c3d_mask):
            start_id = start_ids[ind]
            end_id = end_ids[ind]
            c3d_feats =video_feat_sequence[start_id:end_id]
            # padding if needed
            clip_len = end_id - start_id
            c3d_feats = tf.cond(tf.less(clip_len, max_clip_len), lambda: tf.concat([c3d_feats, tf.fill([max_clip_len-clip_len, self.options['video_feat_dim']], 0.)], axis=0), lambda: c3d_feats[:max_clip_len])
            c3d_feats = tf.expand_dims(c3d_feats, axis=0)
            event_c3d_sequence = tf.concat([event_c3d_sequence, c3d_feats], axis=0)

            this_mask = tf.cond(tf.less(clip_len, max_clip_len), lambda: tf.concat([tf.fill([clip_len], 1.), tf.fill([max_clip_len-clip_len], 0.)], axis=0), lambda: tf.fill([max_clip_len], 1.))
            this_mask = tf.expand_dims(this_mask, axis=0)
            this_mask = tf.to_int32(this_mask)
            event_c3d_mask = tf.concat([event_c3d_mask, this_mask], axis=0)
            

            return tf.add(ind, 1), event_c3d_sequence, event_c3d_mask

        _, event_c3d_sequence, event_c3d_mask = tf.while_loop(condition, body, loop_vars=[ind, event_c3d_sequence, event_c3d_mask], shape_invariants=[ind.get_shape(), tf.TensorShape([None, None, self.options['video_feat_dim']]), tf.TensorShape([None, None])])


        return event_c3d_sequence, event_c3d_mask
