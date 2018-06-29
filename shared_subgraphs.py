import tensorflow as tf
# from tensorbayes.layers import Constant, Placeholder, Dense, GaussianSample
# from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
# from tensorbayes.tbutils import cross_entropy_with_logits
from tensorbayes.layers import constant as Constant
from tensorbayes.layers import placeholder as Placeholder
from tensorbayes.layers import dense as Dense
from tensorbayes.layers import conv2d as Conv2d
from tensorbayes.layers import max_pool as Max_pool
from tensorbayes.layers import gaussian_sample as GaussianSample
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
from tensorbayes.tbutils import softmax_cross_entropy_with_two_logits as cross_entropy_with_logits
import numpy as np
import sys

# vae subgraphs
# encoder
def qy_graph(x, k=10):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        x = tf.reshape(x,[-1, 28, 28, 1])
        # h1 = Dense(x, 512, 'layer1', tf.nn.relu, reuse=reuse)
        # h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        h1 = Conv2d(x, 32, [3,3], [1,1], scope = 'convlayer1', activation = tf.nn.relu, reuse = reuse)
        h1 = Max_pool(h1, [2,2], [1,1], scope = 'convlayer1')
        h2 = Conv2d(h1, 32, [3,3], [1,1], activation = tf.nn.relu, reuse = reuse, scope = 'convlayer2')
        h2 = Max_pool(h2, [2,2], [1,1], scope = 'convlayer2')
        h2_flat = tf.contrib.layers.flatten(h2, scope = "flatten")
        qy_logit = Dense(h2_flat, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        x = tf.reshape(x, [-1, 28, 28, 1])
        h1_x = h1 = Conv2d(x, 32, [3,3], [1,1], scope = 'convlayer1', activation = tf.nn.relu, reuse = reuse)
        h1_x = Max_pool(h1_x, [2,2], [1,1], scope = 'convlayer1')
        h2_x = Conv2d(h1_x, 32, [3,3], [1,1], activation = tf.nn.relu, reuse = reuse, scope = 'convlayer2')
        h2_x = Max_pool(h2_x, [2,2], [1,1], scope = 'convlayer2')
        h2_x_flat = tf.contrib.layers.flatten(h2_x,scope = "flatten")
        xy = tf.concat((y, h2_x_flat),1, name='xy/concat')
        h1 = Dense(xy, 256, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 256, 'layer2', tf.nn.relu, reuse=reuse)
        zm = Dense(h2, 64, 'zm', reuse=reuse)
        zv = Dense(h2, 64, 'zv', tf.nn.softplus, reuse=reuse)
        z = GaussianSample(zm, zv, 'z')
    return z, zm, zv

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)
