import tensorflow as tf
import numpy as np
from models import Hred
from dialogue import Dialogue


def reply(model, sess, sentences):
    sentence_len = len(sentences)
    tokens_arr = [dialogue.tokenizer(sentence) for sentence in sentences]
    max_len = max([len(seq) for seq in tokens_arr])

    padded = [dialogue.pad(tokens, max_len) for tokens in tokens_arr]
    ids_arr = [dialogue.tokens_to_ids(tokens) for tokens in padded]
    result = sess.run(model.sample_id,
                      feed_dict={model.source: ids_arr,
                                 model.source_seq_length: [len(ids) for ids in ids_arr],
                                 model.num_utterance: sentence_len})
    result = result[-1]
    end = np.where(result == 2)

    reply = ' '.join(dialogue.ids_to_tokens(result[:end]))
    if reply == '':
        reply = '.....'
    return reply


def chat(model):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./model')
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    line = input(">")
    sentences = []

    while line:
        sentences.append(line)
        response = reply(model, sess, sentences)
        sentences.append(response)
        line = input(response + "\n>")


path = './data/conversation.txt'
dialogue = Dialogue(path)

hparams = tf.contrib.training.HParams(total_epochs=1000,
                                      num_units=128,
                                      learning_rate=0.0001,
                                      voc_size=dialogue.voc_size,
                                      embedding_size=100,
                                      total_batch=len(dialogue.dialogues))

model = Hred(hparams, 'infer')
chat(model)


