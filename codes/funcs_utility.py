import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def distance(f, w, value=0):
    if value == 0:
        f_expand = tf.expand_dims(tf.expand_dims(f, axis=1), axis=1)
        w_expand = tf.expand_dims(w, axis=1)
        fw = f_expand - w_expand
        dist = fw ** 2
        dist = tf.reduce_sum(tf.squeeze(dist), -1)
    else:
        f_expand = np.expand_dims(np.expand_dims(f, axis=1), axis=1)
        w_expand = np.expand_dims(w, axis=1)
        fw = f_expand - w_expand
        dist = fw ** 2
        dist = np.sum(np.squeeze(dist), -1)

    return dist


def plot_feature(features, y, center, region, class_num, center_num, name):
    """
    Plot the trained occ region in the feature space
    """
    plt.scatter(features[:, 0], features[:, 1], c=y, cmap=plt.cm.tab10)
    plt.scatter(center[:, 0], center[:, 1], s=20, c='black')
    theta = np.arange(0, 2 * np.pi, 0.01)
    if region is not None:
        for i in range(class_num * center_num):
            xc = center[i, 0] + (region[0, i]) ** 0.5 * np.cos(theta)
            yc = center[i, 1] + (region[0, i]) ** 0.5 * np.sin(theta)
            plt.plot(xc, yc, c='black')
    if name == 'train':
        plt.title("latent space of training data")
    if name == 'test':
        plt.title("latent space of testing data")
    plt.show()
