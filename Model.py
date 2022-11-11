# coding:utf-8
import tensorflow as tf
import seq_loss


class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate

        self._input_data = tf.compat.v1.placeholder(
            tf.int32, [batch_size, num_steps])
        self._targets = tf.compat.v1.placeholder(
            tf.int32, [batch_size, num_steps])  # 宣告輸入變數 x, y

        lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            # size 是 wordembedding 的維度
            embedding = tf.compat.v1.get_variable(
                "embedding", [vocab_size, size])
            # 回傳一個 tensor，shape 是 (batch_size, num_steps, size)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.compat.v1.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                # inputs[:, time_step, :] 的 shape 是 (batch_size, size)
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        softmax_w = tf.compat.v1.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.compat.v1.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._final_state = state

        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        loss = seq_loss.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op
