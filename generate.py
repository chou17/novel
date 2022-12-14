# coding:utf-8
import tensorflow as tf
import sys
import numpy as np
import pickle as cPickle
import Config
import Model

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()

char_to_idx, idx_to_char = cPickle.load(open(config.model_path+'.voc', 'rb'))

config.vocab_size = len(char_to_idx)
is_sample = config.is_sample
is_beams = config.is_beams
beam_size = config.beam_size
len_of_generation = config.len_of_generation
start_sentence = config.start_sentence
if (len(sys.argv) == 2):
    if (sys.version_info > (3, 0)):
        start_sentence = sys.argv[1]
    else:
        start_sentence = sys.argv[1].decode("utf-8")


def run_epoch(session, m, data, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                                  {m.input_data: x,
                                   m.initial_state: state})
    return prob, _state


def main(_):
    with tf.Graph().as_default(), tf.compat.v1.Session(config=config_tf) as session:
        config.batch_size = 1
        config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.compat.v1.variable_scope("model", reuse=None, initializer=initializer):
            mtest = Model.Model(is_training=False, config=config)

        # tf.global_variables_initializer().run()

        model_saver = tf.compat.v1.train.Saver()
        print("model loading ...")
        model_saver.restore(session, config.model_path +
                            '-%d' % config.save_time)
        print("Done!")

        if not is_beams:
            # sentence state
            char_list = list(start_sentence)
            start_idx = char_to_idx[char_list[0]]
            _state = mtest.initial_state.eval()
            test_data = np.int32([start_idx])
            prob, _state = run_epoch(
                session, mtest, test_data, tf.no_op(), _state)
            gen_res = [char_list[0]]
            for i in range(1, len(char_list)):
                char = char_list[i]
                try:
                    char_index = char_to_idx[char]
                except KeyError:
                    char_index = np.argmax(prob.reshape(-1))
                prob, _state = run_epoch(session, mtest, np.int32(
                    [char_index]), tf.no_op(), _state)
                gen_res.append(char)
            # gen text
            if is_sample:
                gen = np.random.choice(
                    config.vocab_size, 1, p=prob.reshape(-1))
                gen = gen[0]
            else:
                gen = np.argmax(prob.reshape(-1))
            test_data = np.int32(gen)
            gen_res.append(idx_to_char[gen])
            for i in range(len_of_generation-1):
                prob, _state = run_epoch(
                    session, mtest, test_data, tf.no_op(), _state)
                if is_sample:
                    gen = np.random.choice(
                        config.vocab_size, 1, p=prob.reshape(-1))
                    gen = gen[0]
                else:
                    gen = np.argmax(prob.reshape(-1))
                test_data = np.int32(gen)
                gen_res.append(idx_to_char[gen])
            print("Generated Result: {0}".format(gen_res))
            str_result = "".join(gen_res)
            path = 'output.txt'
            f = open(path, 'w')
            f.write(str_result)
            f.close()

        else:
            # sentence state
            char_list = list(start_sentence)
            start_idx = char_to_idx[char_list[0]]
            _state = mtest.initial_state.eval()
            beams = [(0.0, [idx_to_char[start_idx]], idx_to_char[start_idx])]
            test_data = np.int32([start_idx])
            prob, _state = run_epoch(
                session, mtest, test_data, tf.no_op(), _state)
            y1 = np.log(1e-20 + prob.reshape(-1))
            beams = [(beams[0][0], beams[0][1], beams[0][2], _state)]
            for i in range(1, len(char_list)):
                char = char_list[i]
                try:
                    char_index = char_to_idx[char]
                except KeyError:
                    top_indices = np.argsort(-y1)
                    char_index = top_indices[0]
                prob, _state = run_epoch(session, mtest, np.int32(
                    [char_index]), tf.no_op(), beams[0][3])
                y1 = np.log(1e-20 + prob.reshape(-1))
                beams = [(beams[0][0], beams[0][1] +
                          [char], char_index, _state)]
            # gen text
            if is_sample:
                top_indices = np.random.choice(
                    config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
            else:
                top_indices = np.argsort(-y1)
            b = beams[0]
            beam_candidates = []
            for i in range(beam_size):
                wordix = top_indices[i]
                beam_candidates.append(
                    (b[0] + y1[wordix], b[1] + [idx_to_char[wordix]], wordix, _state))
            # decreasing order
            beam_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = beam_candidates[:beam_size]  # truncate to get new beams
            for xy in range(len_of_generation-1):
                beam_candidates = []
                for b in beams:
                    test_data = np.int32(b[2])
                    prob, _state = run_epoch(
                        session, mtest, test_data, tf.no_op(), b[3])
                    y1 = np.log(1e-20 + prob.reshape(-1))
                    if is_sample:
                        top_indices = np.random.choice(
                            config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
                    else:
                        top_indices = np.argsort(-y1)
                    for i in range(beam_size):
                        wordix = top_indices[i]
                        beam_candidates.append(
                            (b[0] + y1[wordix], b[1] + [idx_to_char[wordix]], wordix, _state))
                # decreasing order
                beam_candidates.sort(key=lambda x: x[0], reverse=True)
                # truncate to get new beams
                beams = beam_candidates[:beam_size]

            if (sys.version_info > (3, 0)):
                print("Generated Result: {0}".format(beams[0][1]))
                str_result = "".join(beams[0][1])
                path = 'output.txt'
                f = open(path, 'w')
                f.write(str_result)
                f.close()
            else:
                print('Generated Result:')
                print(''.join(beams[0][1]))


if __name__ == "__main__":
    tf.compat.v1.app.run()
