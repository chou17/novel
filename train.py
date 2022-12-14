# coding:utf-8
import tensorflow as tf
import sys
import time
import numpy as np
import pickle as cPickle
import os
import Config
import Model


if not os.path.exists('./model'):
    os.makedirs('./model')

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

file = sys.argv[1]
if (sys.version_info > (3, 0)):
    data = open(file, encoding="utf-8").read()
else:
    data = open(file, 'r').read()
    data = data.decode('utf-8')

chars = list(set(data))  # char vocabulary

data_size, _vocab_size = len(data), len(chars)
print("data has {0} characters, {1} unique.".format(data_size, _vocab_size))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

config = Config.Config()
config.vocab_size = _vocab_size

cPickle.dump((char_to_idx, idx_to_char), open(
    config.model_path+'.voc', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

context_of_idx = [char_to_idx[ch] for ch in data]


def data_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        # data 的 shape 是 (batch_size, batch_len)，每一行是連貫的一段，一次可輸入多個段落
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]  # y 就是 x 的錯一位，即下一個詞
        yield (x, y)


def run_epoch(session, m, data, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(data_iterator(data, m.batch_size,
                                                m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],  # x 和 y 的 shape 都是 (batch_size, num_steps)
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if step and step % (epoch_size // 10) == 0:
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   (time.time() - start_time)))
            start_time = time.time()

    return np.exp(costs / iters)


def main(_):
    train_data = context_of_idx

    with tf.Graph().as_default(), tf.compat.v1.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.compat.v1.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(is_training=True, config=config)

        tf.compat.v1.global_variables_initializer().run()

        model_saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

        for i in range(config.iteration):
            print("Training Epoch: %d ..." % (i+1))
            train_perplexity = run_epoch(session, m, train_data, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f" %
                  (i + 1, train_perplexity))

            if (i+1) % config.save_freq == 0:
                print("model saving ...")
                model_saver.save(session, config.model_path+'-%d' % (i+1))
                print("Done!")


if __name__ == "__main__":
    tf.compat.v1.app.run()
