import numpy as np
import tensorflow as tf
from funcs_utility import distance


# distance based cross entropy loss (DCE)
def dce_loss(y_batch, probs):
    batch_size = len(y_batch)
    class_num = np.shape(probs)[-1]
    y_batch_one = tf.one_hot(y_batch, depth=class_num)

    eps = np.zeros(tf.shape(probs))
    idx = tf.where(probs == 0)
    eps[idx[:, 0], idx[:, 1]] = np.finfo(float).eps
    ce = -tf.math.log(probs + eps)
    loss = ce * y_batch_one
    loss = tf.reduce_sum(loss) / batch_size

    return loss


def w2pl_loss(y_batch, dist, alpha):
    # normalize alpha
    alpha = alpha / np.mean(alpha)

    batch_size = len(y_batch)
    y_reshape = tf.reshape(y_batch, [-1, 1])
    line = np.arange(batch_size).reshape(-1, 1)
    index = np.hstack([line, y_reshape])

    batch_alpha = alpha[0, y_batch]
    batch_dist = tf.gather_nd(dist, index) * batch_alpha
    loss = tf.reduce_mean(batch_dist)

    return loss


def region_loss(y, region, dist_fp, alpha, ratio):
    # normalize alpha
    alpha = alpha / np.min(alpha)

    class_num = np.shape(alpha)[-1]
    batch_size = len(y)
    batch_alpha = alpha[0, y]
    batch_alpha = np.expand_dims(batch_alpha, -1)

    filter_y = np.zeros([batch_size, class_num])
    filter_y[np.arange(batch_size), y] = 1

    radius = region ** 0.5
    loss_in = (dist_fp**0.5 - radius) * filter_y
    filter_in = np.ones([len(y), class_num])
    idx = tf.where(loss_in <= 0)
    filter_in[idx[:, 0], idx[:, 1]] = 0
    loss_in = loss_in * filter_in
    loss_in = tf.reduce_sum(loss_in**2 * batch_alpha) / batch_size

    loss_R = tf.reduce_sum(region * filter_y * batch_alpha) / batch_size

    loss = loss_in + loss_R * ratio

    return loss


def compute_probs(y, dist, margin, alpha, tem_scale):
    # min_alpha = 1
    alpha = alpha / np.min(alpha)

    class_num = np.shape(alpha)[-1]
    batch_size = len(y)
    batch_margin = margin[0, y]
    batch_margin = np.expand_dims(batch_margin, 1)

    dist_self = dist
    filter_self = np.zeros([batch_size, class_num])
    filter_self[np.arange(batch_size), y] = 1
    dist_self = dist_self * filter_self

    dist_other = dist - batch_margin
    filter_other = np.ones([batch_size, class_num])
    filter_other[np.arange(batch_size), y] = 0
    logits = - (dist_self + dist_other * filter_other) / tem_scale

    base_probs = 1 / alpha
    base_probs = base_probs / np.sum(base_probs)
    logits = loss_fn_logits(logits, base_probs, tau=1.0)

    probs = tf.nn.softmax(logits)

    return probs


def compute_condition_probs(prototypes, margin, tem_scale):
    dist_pp = distance(prototypes, prototypes)
    filter = np.eye(dist_pp.shape[0])
    filter = 1 - filter
    logits = - (dist_pp - margin * filter) / tem_scale

    probs_yc = tf.nn.softmax(logits)  # 10x10
    probs_yc = tf.linalg.diag_part(probs_yc)

    return probs_yc


def my_final_loss(features, prototypes, region, y_batch, alpha, tem_scale=1):
    ratio = 0.05
    alpha = alpha/np.min(alpha)

    proto_value = np.zeros(tf.shape(prototypes))
    for i in range(tf.shape(prototypes)[0]):
        proto_value[i, :] = tf.get_static_value(prototypes[i, :])
    dist_pp_value = distance(proto_value, proto_value, value=1)
    region_pp = np.expand_dims(np.sort(dist_pp_value, 0)[1, :] / 4, 0)
    region_value = tf.get_static_value(region)
    region_pp_max = np.max(region_pp) * np.ones(np.shape(region_value))
    region_pp_min = np.min(region_pp) * np.ones(np.shape(region_value))
    region_value_max = np.max(region_value) * np.ones(np.shape(region_value))

    region_confine = region_pp_min
    region_largest = np.max(np.r_[region_pp_max, region_value_max], 0)
    region_largest = np.expand_dims(region_largest, 0)
    beta = np.min(np.r_[2*np.ones(np.shape(region_value)), region_largest / region_confine], 0)
    beta = np.expand_dims(beta, 0)

    margin = region_largest - region_confine
    margin = np.min(np.r_[region_confine, margin], 0)
    margin = np.expand_dims(margin, 0)

    probs_yc = compute_condition_probs(prototypes, margin, tem_scale)
    loss_yc = tf.reduce_mean(-tf.math.log(probs_yc))

    new_features = features
    new_dist = distance(new_features, prototypes)

    dist_region = tf.get_static_value(new_dist)
    loss_region = region_loss(y_batch, region, dist_region, alpha, ratio)

    probs = compute_probs(y_batch, new_dist, margin, alpha, tem_scale)
    loss_ce = dce_loss(y_batch, probs)
    loss_pl = w2pl_loss(y_batch, new_dist, alpha)

    loss = loss_ce + loss_region + loss_pl*(beta[0, 0]-1) + loss_yc

    return loss, beta, margin, probs_yc


def loss_fn_logits(logits, base_probs, tau=1.0):
    logits = logits + tf.math.log(
          tf.cast(base_probs**tau + 1e-12, dtype=tf.float32))
    return logits
