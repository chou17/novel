from six.moves import zip  # pylint: disable-redefined
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                        logits+targets+weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    labels=target, logits=logit)
            else:
                crossent = softmax_loss_function(labels=target, logits=logit)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by for all-weights.
            log_perps /= total_size
    return log_perps
