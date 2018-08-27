import tensorflow as tf


class Seq2Seq(object):

    def __init__(self, hparams, mode):
        self.mode = mode
        self.embeddings = tf.Variable(tf.random_uniform([hparams.voc_size, hparams.embedding_size], -1.0, 1.0))

        self.source = tf.placeholder(tf.int32, shape=[None, None], name='source')
        self.target_input = tf.placeholder(tf.int32, shape=[None, None], name='target_input')
        self.target_output = tf.placeholder(tf.int32, shape=[None, None], name='target_output')

        self.source_seq_length = tf.placeholder(tf.int32, shape=[None], name='source_seq_length')
        self.target_seq_length = tf.placeholder(tf.int32, shape=[None], name='target_seq_length')

        self.num_utterance = tf.placeholder(tf.int32, shape=[], name='num_utterance')

        self.num_units = hparams.num_units
        self.voc_size = hparams.voc_size

        self.logits, self.loss, self.sample_id = self.build_graph(hparams)

        if mode == 'train':
            self.train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def build_graph(self, hparams):
        _, enc_state = self.build_encoder(hparams)

        logits, sample_id = self.build_decoder(hparams, enc_state)

        loss = self.compute_loss(logits)

        return logits, loss, sample_id

    def build_encoder(self, hparams):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            enc_emb_input = tf.nn.embedding_lookup(self.embeddings, self.source)
            cell = self.build_encoder_cell(hparams)
            outputs, state = tf.nn.dynamic_rnn(cell, enc_emb_input, self.source_seq_length, dtype=tf.float32)

        return outputs, state

    def build_encoder_cell(self, hparams):
        cell = tf.nn.rnn_cell.GRUCell(hparams.num_units)
        return cell

    def build_decoder(self, hparams, initial_state):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

            cell = self.build_encoder_cell(hparams)
            dec_emb_input = tf.nn.embedding_lookup(self.embeddings, self.target_input)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                helper = tf.contrib.seq2seq.TrainingHelper(dec_emb_input, self.target_seq_length)
            else:
                start_tokens = tf.fill([self.num_utterance], 1)
                end_token = 2
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_token)

            # initial_state shape check.!!!!
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state,
                                                      output_layer=tf.layers.Dense(hparams.voc_size, use_bias=False))

            outputs, states, length = tf.contrib.seq2seq.dynamic_decode(decoder)

            logits = outputs.rnn_output
            sample_id = outputs.sample_id

        return logits, sample_id

    def build_decoder_cell(self, hparams):
        cell = tf.nn.rnn_cell.GRUCell(hparams.num_units)
        return cell

    def compute_loss(self, logits):
        max_time = self.get_max_time(self.target_output)
        weights = tf.sequence_mask(self.target_seq_length, max_time, dtype=logits.dtype)
        loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits, self.target_output, weights))

        return loss

    def get_max_time(self, tensor):

        return tensor.shape[1].value


class Hred(Seq2Seq):
    def build_graph(self, hparams):
        _, enc_state = self.build_encoder(hparams)

        context_input = tf.reshape(enc_state, shape=[1, -1, hparams.num_units])

        context_output = self.build_context(hparams, context_input)

        initial_state = tf.reshape(context_output, shape=[-1, hparams.num_units])

        logits, sample_id = self.build_decoder(hparams, initial_state)

        loss = self.compute_loss(logits)

        return logits, loss, sample_id

    def build_context_cell(self, hparams):
        cell = tf.nn.rnn_cell.GRUCell(hparams.num_units)
        return cell

    def build_context(self, hparams, context_input):
        with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
            cell = self.build_context_cell(hparams)
            initial_state = cell.zero_state(context_input.get_shape()[0], dtype=tf.float32)
            context_output, _ = tf.nn.dynamic_rnn(cell, context_input, initial_state=initial_state)

        return context_output

