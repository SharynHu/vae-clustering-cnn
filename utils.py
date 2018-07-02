from tensorbayes.utils import progbar
from scipy.stats import mode
import numpy as np
import os.path

def stream_print(f, string, pipe_to_file=True):
    print string
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def test_acc(mnist, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    # for i in range(10):
    #     print logits[1000 * i + 300]

    cat_pred = logits.argmax(1)
    # print cat_pred[:20]
    # print cat_pred[1000:1020]
    # print cat_pred[2000:2020]
    # print cat_pred[3000:3020]
    # raise ValueError('Nothing')
    # print cat_pred
    real_pred = np.zeros_like(cat_pred)
    for cat in xrange(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
        # print real_pred
    return np.mean(real_pred == mnist.test.labels.argmax(1))

def open_file(fname):
    if fname is None:
        return None
    else:
        i = 0
        while os.path.isfile('{:s}.{:d}'.format(fname, i)):
            i += 1
        return open('{:s}.{:d}'.format(fname, i), 'w', 0)

def train(fname, mnist, sess_info, epochs):
    (sess, qy_logit, nent, loss, train_step) = sess_info
    f = open_file(fname)
    iterep = 500
    for i in range(iterep * epochs):
        sess.run(train_step, feed_dict={'x:0': mnist.train.next_batch(100)[0]})
        # if i<= 20:
        #     print test_acc(mnist, sess, qy_logit)
        # else:
        #     raise ValueError("nothing")
        # a, b = sess.run([nent, loss], feed_dict={'x:0': mnist.train.images[np.random.choice(50000, 10000)]})
        # c, d = sess.run([nent, loss], feed_dict={'x:0': mnist.test.images})
        # a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
        # e = test_acc(mnist, sess, qy_logit)
        # string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
        #             .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'iteration')+'\n')
        # stream_print(f, string, i <= iterep)
        # string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
        #             .format(a, b, c, d, e, i + 1))
        # stream_print(f, string)
        progbar(i, iterep)
        # print '\n'
        if (i + 1) %  iterep == 0:
            a, b = sess.run([nent, loss], feed_dict={'x:0': mnist.train.images[np.random.choice(50000, 10000)]})
            c, d = sess.run([nent, loss], feed_dict={'x:0': mnist.test.images})
            a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(mnist, sess, qy_logit)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i <= iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, (i + 1) / iterep))
            stream_print(f, string)
    if f is not None: f.close()
