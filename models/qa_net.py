import numpy as np
import tensorflow as tf
import sys, os, math
from tensorflow.python.framework import function

# some methods modified from: https://github.com/openai/finetune-transformer-lm

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    return x

def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

def get_ema_if_exists(v, gvs):
    name = v.name.split(':')[0]
    ema_name = name+'/ExponentialMovingAverage:0'
    ema_v = [v for v in gvs if v.name == ema_name]
    if len(ema_v) == 0:
        ema_v = [v]
    return ema_v[0]

def get_ema_vars(*vs):
    if tf.get_variable_scope().reuse:
        gvs = tf.global_variables()
        vs = [get_ema_if_exists(v, gvs) for v in vs]
    if len(vs) == 1:
        return vs[0]
    else:
        return vs

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, attn_pdrop, mask=True, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))
    if mask:
        w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), 
    pad='VALID', train=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, attn_pdrop, resid_pdrop, mask=True, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, attn_pdrop, mask=mask, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, resid_pdrop, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        h = gelu(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    return e

def layer_dropout(x, res, pdrop, train):
    if train and pdrop > 0:
        return tf.cond(tf.random_uniform([]) < pdrop, lambda:res, lambda:tf.nn.dropout(x, 1-pdrop)+res)
    return x+res

def highway(x, scope, nf, pdrop, n_layers=2, train=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x = conv1d(x, "c_proj", nf, 1, train=train)
        for i in range(n_layers):
            t = tf.nn.sigmoid(conv1d(x, "gate_%d"%i, nf, 1, train=train))
            h = conv1d(x, "act_%d"%i, nf, 1, train=train)
            h = dropout(h, pdrop, train)
            x = h * t + x * (1 - t)
        return x

def pos_emb(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    return signal

def depthwise_separable_convolution(x, scope, fh, fw, nf, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        df = tf.get_variable("df", [fh, fw, nx, 1])
        pf = tf.get_variable("pf", [1, 1, nx, nf])
        b = tf.get_variable("b", nf)
        c = tf.nn.separable_conv2d(x, df, pf, strides=[1, 1, 1, 1], padding="SAME")+b
        c = tf.nn.relu(c)
        return c

def conv_block(x, scope, n_conv, fw, nf, pdrop, l, L, train=False):
    with tf.variable_scope(scope):
        x = tf.expand_dims(x, 2)
        for i in range(n_conv):
            n = norm(x, scope="ln_%d"%i)
            if i % 2 == 0:
                n = dropout(n, pdrop, train)
            n = depthwise_separable_convolution(n, "depth_conv_%d"%i, fw, 1, nf, train)
            x = layer_dropout(n, x, pdrop * float(l) / L, train)
            l += 1
        x = tf.squeeze(x, 2)
        return x, l

def res_block(x, scope, n_block, n_conv, fw, units, n_head, attn_pdrop, resid_pdrop, train=False, scale=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        nx = shape_list(x)[-1]
        h = x + pos_emb(*shape_list(x)[-2:])
        l = 1
        L = (n_conv + 2) * n_block
        for i in range(n_block):
            c, l = conv_block(h, "c_block_%d"%i, n_conv, fw, units, resid_pdrop, l, L, train=train)
            a = attn(c, 'attn_%d'%i, nx, n_head, attn_pdrop, resid_pdrop, mask=False, train=train, scale=scale)
            n = norm(c+a, 'ln_%d'%i)
            h = mlp(n, 'mlp_%d'%i, nx*4, resid_pdrop, train=train)
        return h

def trilinear(c, q, scope, pdrop, train=False):
    with tf.variable_scope(scope):
        # c . Wc + (q . Wq)_T + (c*w) . q_T
        nu = shape_list(c)[-1]
        Wc = tf.get_variable("Wc", [nu, 1])
        Wq = tf.get_variable("Wq", [nu, 1])
        W = tf.get_variable("W", [nu])
        c = dropout(c, pdrop, train)
        q = dropout(q, pdrop, train)
        cWc = tf.tensordot(c, Wc,[[2],[0]])
        qWq = tf.transpose(tf.tensordot(q, Wq,[[2],[0]]), [0,2,1])
        cWq = tf.matmul(c*W, q, transpose_b=True)
        s = cWc + qWq + cWq
        return s

def ctq_attn(c, q, scope, pdrop, train=False):
    with tf.variable_scope(scope):
        s = trilinear(c, q, 'trilinear', pdrop)
        s_ = tf.nn.softmax(s, 0)
        s__ = tf.nn.softmax(s, 1)
        a = tf.matmul(s_, q)
        b = tf.matmul(tf.matmul(s_, s__, transpose_b=True), c)
        attn = [c, a, c * a, c * b]
        return attn

def qa_net(Cw, Qw, Cch, Qch, Ys, Ye, n_word, n_char, n_pred=1, n_wembd=300, n_cembd=200, units=128, embd_pdrop=0.1, n_head=12, 
    attn_pdrop=0.1, resid_pdrop=0.1, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        emb_unk = tf.get_variable("emb_unk", [n_wembd], initializer=tf.random_normal_initializer(stddev=0.02))
        emb_w = tf.get_variable("emb_w", [n_word, n_wembd], trainable=False)
        emb_c = tf.get_variable("emb_c", [n_char, n_cembd], initializer=tf.random_normal_initializer(stddev=0.02)) 
        emb_w = tf.concat([tf.expand_dims(emb_unk, 0), emb_w], 0)
        # word embeddings
        Cw = dropout(embed(Cw, emb_w), embd_pdrop, train)
        Qw = dropout(embed(Qw, emb_w), embd_pdrop, train)
        # character embeddings, share weights between convolutions
        Cch = dropout(embed(tf.reshape(Cch, [-1, shape_list(Cch)[-1]]), emb_c), embd_pdrop/2, train)
        Qch = dropout(embed(tf.reshape(Qch, [-1, shape_list(Qch)[-1]]), emb_c), embd_pdrop/2, train)
        Qch = tf.reshape(tf.reduce_max(conv1d(Qch, "emb_conv", n_cembd, 5), 1), shape_list(Qw)[:2]+[-1])
        Cch = tf.reshape(tf.reduce_max(conv1d(Cch, "emb_conv", n_cembd, 5, reuse=True), 1), shape_list(Cw)[:2]+[-1])
        # concat word and character embeddings and pass through a highway network (share weights)
        C = highway(tf.concat([Cw, Cch], 2), "emb_hwy", units, resid_pdrop, train=train)
        Q = highway(tf.concat([Qw, Qch], 2), "emb_hwy", units, resid_pdrop, train=train, reuse=True)
        # pass context and query through residual blocks (share weights)
        C = res_block(C, "emb_enc", 1, 4, 7, units, n_head, attn_pdrop, resid_pdrop, train=train, scale=True)
        Q = res_block(Q, "emb_enc", 1, 4, 7, units, n_head, attn_pdrop, resid_pdrop, train=train, scale=True, reuse=True)
        # pass outputs to context-to-query attention followed by 1d convolution
        attn = ctq_attn(C, Q, "ctq_attn", attn_pdrop, train=train)
        h = conv1d(tf.concat(attn, -1), "", units, 1)
        # triple stacked residual blocks (share weights between residual blocks)
        enc_1 = res_block(h, "stacked_enc", 7, 2, 5, units, n_head, attn_pdrop, resid_pdrop, train=train, scale=True)
        enc_2 = res_block(enc_1, "stacked_enc", 7, 2, 5, units, n_head, attn_pdrop, resid_pdrop, train=train, scale=True, reuse=True)
        enc_3 = res_block(enc_2, "stacked_enc", 7, 2, 5, units, n_head, attn_pdrop, resid_pdrop, train=train, scale=True, reuse=True)
        # get logits and calc loss
        s_logits = tf.squeeze(conv1d(tf.concat([enc_1, enc_2], 2), "s_proj", 1, 1), 2)
        e_logits = tf.squeeze(conv1d(tf.concat([enc_1, enc_3], 2), "e_proj", 1, 1), 2)
        s_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s_logits, labels=Ys)
        e_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e_logits, labels=Ye)
        losses = s_losses + e_losses
        # get predictions
        preds = tf.matmul(tf.expand_dims(tf.nn.softmax(s_logits), axis=2), tf.expand_dims(tf.nn.softmax(e_logits), axis=1))
        preds = tf.matrix_band_part(preds, 0, n_pred)
        s_preds, e_preds = tf.argmax(tf.reduce_max(preds, 2), 1), tf.argmax(tf.reduce_max(preds, 1), 1)
        return s_preds, e_preds, losses