import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
# from tensorbayes.layers import Constant, Placeholder, Dense, GaussianSample
# from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
# from tensorbayes.tbutils import cross_entropy_with_logits
from tensorbayes.layers import constant as Constant
from tensorbayes.layers import placeholder as Placeholder
from tensorbayes.layers import dense as Dense
from tensorbayes.layers import conv2d_transpose as Conv2d_transpose
from tensorbayes.layers import conv2d as Conv2d
from tensorbayes.layers import gaussian_sample as GaussianSample
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as cross_entropy_with_logits
from tensorbayes.nbutils import show_graph
from tensorbayes.utils import progbar
import numpy as np
import sys
from shared_subgraphs import qy_graph, qz_graph, labeled_loss
from utils import train
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def px_graph(z, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
    # -- p(x)
    with tf.variable_scope('px'):
        zy = tf.concat((z, y), 1, name='zy/concat')
        # h1 = Dense(zy, 512, 'layer1', tf.nn.relu, reuse=reuse)
        # h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        # px_logit = Dense(h2, 784, 'logit', reuse=reuse)
        h3 = Dense(zy, 28 * 14 * 14, 'layer3', tf.nn.relu, reuse = reuse )
        h3 = tf.reshape(h3,[-1, 14, 14, 28])
        h4 = Conv2d_transpose(h3, 28, [3, 3], [1, 1], activation=tf.nn.relu, reuse = reuse, scope = "layer4")
        h5 = Conv2d_transpose(h4, 28, [3, 3], [1, 1], activation=tf.nn.relu, reuse = reuse, scope = "layer5")
        h6 = Conv2d_transpose(h5, 28, [3, 3], [2, 2], activation=tf.nn.relu, reuse = reuse, scope = "layer6")
        # h7 = Conv2d_transpose(h6, 28, [3, 3], [1, 1], activation=tf.nn.relu, reuse = reuse, scope = "layer7")
        px_logit = Conv2d(h6, 1, [2, 2], [1, 1] ,scope = "layer7", reuse = reuse)
        px_logit = tf.contrib.layers.flatten(px_logit)
    return px_logit

tf.reset_default_graph()
x = Placeholder((None, 784), name='x')

# binarize data and create a y "placeholder"
with tf.name_scope('x_binarized'):
    xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
with tf.name_scope('y_'):
    y_ = tf.fill(tf.stack([tf.shape(x)[0], 10]), 0.0)

# propose distribution over y
qy_logit, qy = qy_graph(xb)

# for each proposed y, infer z and reconstruct x
z, zm, zv, px_logit = [[None] * 10 for i in xrange(4)]
for i in xrange(10):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        y = tf.add(y_, Constant(np.eye(10)[i], name='hot_at_{:d}'.format(i)))
        z[i], zm[i], zv[i] = qz_graph(xb, y)
        px_logit[i] = px_graph(z[i], y)

# Aggressive name scoping for pretty graph visualization :P
with tf.name_scope('loss'):
    with tf.name_scope('neg_entropy'):
        nent = -cross_entropy_with_logits(logits = qy_logit, labels =qy)
    losses = [None] * 10
    for i in xrange(10):
        with tf.name_scope('loss_at{:d}'.format(i)):
            losses[i] = labeled_loss(xb, px_logit[i], z[i], zm[i], zv[i], Constant(0), Constant(1))
    with tf.name_scope('final_loss'):
        loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in xrange(10)])

train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer()) # Change initialization protocol depending on tensorflow version
# x_run = sess.run(xb,feed_dict = {'x:0': mnist.train.next_batch(100)[0]})
# x_run = x_run.reshape([-1, 28,28])
# im = Image.fromarray(x_run[1]*255)
# im.show()
# variable_name = [c.name for c in tf.trainable_variables()]
# print(variable_name)
# raise ValueError('Nothing')
# weight_in_y_logits = sess.run(u'qy/layer1/weights:0',feed_dict = {'x:0': mnist.train.next_batch(100)[0]})
# bias_in_y_logits = sess.run(u'qy/layer1/biases:0',feed_dict = {'x:0': mnist.train.next_batch(100)[0]})
# print weight_in_y_logits
# print bias_in_y_logits
# raise ValueError('Nothing')
sess_info = (sess, qy_logit, nent, loss, train_step)
# train('logs/m2.log', mnist, sess_info, epochs=1000)
train('logs/m2.log', mnist, sess_info, epochs=20)
