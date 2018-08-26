import tensorflow as tf
from models import Hred
from dialogue import Dialogue


def train(model, hparams):

    with tf.Session() as sess:
        checkpoint = tf.trian.get_checkpint_state('./model')
        if checkpoint and checkpoint.model_checkpoint_path:
            model.saver.restore(sess, checkpoint.model_checkpoint_path)

        for epoch in range(hparams.epochs):
            epoch_loss = 0
            for batch in range(hparams.total_batch):
                enc_batch, dec_batch, target_batch, enc_seq_len, dec_seq_len, max_len = dialogue.next_dialogue()
                _cost, _ = sess.run([model.loss, model.train_op], feed_dict={model.source: enc_batch,
                                                                                         model.target_input: dec_batch,
                                                                                         model.target_output: target_batch,
                                                                                         model.source_seq_length: enc_seq_len,
                                                                                         model.target_seq_length: dec_seq_len})

                #     print(_cost, _a)
                epoch_loss += _cost / hparams.total_batch
            if epoch % 100 == 0:
                model.saver.saver(sess, './model/hred', global_step=epoch)


path = './data/koen.txt'
dialogue = Dialogue(path)

hparams = tf.contrib.training.HParams(total_epochs=1000,
                                      num_units=128,
                                      learning_rate=0.0001,
                                      voc_size=dialogue.voc_size,
                                      embedding_size=100,
                                      total_batch=len(dialogue.dialogues))

embeddings = tf.Variable(tf.random_uniform([hparams.voc_size, hparams.embedding_size], -1.0, 1.0))
train_model = Hred(hparams, 'train', embeddings)
print("start")
train(train_model, hparams)

