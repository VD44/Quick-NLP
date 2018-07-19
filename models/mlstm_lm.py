import tensorflow as tf
from tensorflow.python.framework import function

# modified from: https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/encoder.py

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    return x

def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    return e

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mlstm_cell(x, c, h, scope, units, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0)):
    with tf.variable_scope(scope):
        x_dim = shape_list(x)[1]
        wx = tf.get_variable("wx", [x_dim, units * 4], initializer=w_init)
        wh = tf.get_variable("wh", [units, units * 4], initializer=w_init)
        wmx = tf.get_variable("wmx", [x_dim, units], initializer=w_init)
        wmh = tf.get_variable("wmh", [units, units], initializer=w_init)
        gx = tf.get_variable("gx", [units * 4], initializer=w_init)
        gh = tf.get_variable("gh", [units * 4], initializer=w_init)
        gmx = tf.get_variable("gmx", [units], initializer=w_init)
        gmh = tf.get_variable("gmh", [units], initializer=w_init)
        b = tf.get_variable("b", [units * 4], initializer=b_init)
        wx = tf.nn.l2_normalize(wx, axis=0) * gx
        wh = tf.nn.l2_normalize(wh, axis=0) * gh
        wmx = tf.nn.l2_normalize(wmx, axis=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, axis=0) * gmh

        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b

        i, f, o, u = tf.split(z, 4, 1)
        c = tf.nn.sigmoid(f) * c + tf.nn.sigmoid(i) * tf.tanh(u)
        h = tf.nn.sigmoid(o) * tf.tanh(c)
        return h, c

def language_mlstm(X, M, units, n_vocab, n_special=2, n_embd=768, embd_pdrop=0.1, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse) as scope:
        we = tf.get_variable("we", [n_vocab+n_special, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        wp = tf.get_variable("w_proj", [units, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        h = tf.zeros([shape_list(X)[0], units])
        c = tf.zeros([shape_list(X)[0], units])

        hs = []
        for idx, x in enumerate(tf.unstack(embed(X, we), axis=1)):
            if idx > 0:
                scope.reuse_variables()
            h, c = mlstm_cell(x, c, h, 'mlstm_cell', units)
            hs.append(h)

        lm_h = tf.reshape(tf.stack(hs, axis=1)[:, :-1], [-1, units])
        lm_h = tf.matmul(lm_h, wp)
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
        return lm_logits, lm_losses