import tensorflow as tf
import numpy as np
import os, time, math, json, joblib, random, argparse

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from util.opt import adam
from util.utils import iter_data, find_trainable_variables, ResultLogger, assign_to_gpu, average_grads, make_path
from data.data import squad_data

from models.qa_net import qa_net as model

# modified from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            s_preds, e_preds, qa_losses = model(*xs, 
                n_word=n_word, n_char=n_char, n_pred=n_pred, n_wembd=n_wembd, n_cembd=n_cembd, units=units, embd_pdrop=embd_pdrop, n_head=n_head, 
                attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, train=True, reuse=do_reuse)
            train_loss = tf.reduce_mean(qa_losses)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([s_preds, e_preds, qa_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    train = adam(params, grads, lr, lr_schedule, n_updates_total, warmup=lr_warmup, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            s_preds, e_preds, qa_losses = model(*xs, 
                n_word=n_word, n_char=n_char, n_pred=n_pred, n_wembd=n_wembd, n_cembd=n_cembd, units=units, embd_pdrop=embd_pdrop, n_head=n_head, 
                attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, train=False, reuse=True)
            gpu_ops.append([s_preds, e_preds, qa_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def iter_apply(Cws, Qws, Cchs, Qchs, Yss, Yes):
    fns = [lambda x:np.concatenate(x, 0), lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for cwmb, qwmb, cchmb, qchmb, ysmb, yemb in iter_data(Cws, Qws, Cchs, Qchs, Yss, Yes, n_batch=n_batch_train, truncate=True, verbose=True):
        n = len(cwmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_s_preds, eval_mgpu_e_preds, eval_mgpu_qa_loss], {Cw_train:cwmb, Qw_train:qwmb, Cch_train:cchmb, Qch_train:qchmb, Ys_train:ysmb, Ye_train:yemb})
        else:
            res = sess.run([eval_s_preds, eval_e_preds, eval_qa_loss], {Cw:cwmb, Qw:qwmb, Cch:cchmb, Qch:qchmb, Ys:ysmb, Ye:yemb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Cws, Qws, Cchs, Qchs):
    s_preds = []
    e_preds = []
    for cwmb, qwmb, cchmb, qchmb in iter_data(Cws, Qws, Cchs, Qchs, n_batch=n_batch_train, truncate=True, verbose=True):
        n = len(cwmb)
        if n == n_batch_train:
            s_p, e_p = sess.run([eval_mgpu_s_preds, eval_mgpu_e_preds], {Cw_train:cwmb, Qw_train:qwmb, Cch_train:cchmb, Qch_train:qchmb})
        else:
            s_p, e_p = sess.run([eval_s_preds, eval_e_preds], {Cw:cwmb, Qw:qwmb, Cch:cchmb, Qch:qchmb})
        s_preds.append(s_p)
        e_preds.append(e_p)
    s_preds = np.concatenate(s_preds, 0)
    e_preds = np.concatenate(e_preds, 0)
    return s_preds, e_preds

def save(path):
    ps = sess.run(params)
    joblib.dump(ps, make_path(path))

def log():
    global best_score
    tr_s_preds, tr_e_preds, tr_cost = iter_apply(trCtxW[:n_valid], trQW[:n_valid], trCtxCh[:n_valid], trQCh[:n_valid], trYs[:n_valid], trYe[:n_valid])
    va_s_preds, va_e_preds, va_cost = iter_apply(vaCtxW, vaQW, vaCtxCh, vaQCh, vaYs, vaYe)
    tr_cost = tr_cost/len(trCtxW[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = (accuracy_score(tr_s_preds, trYs[:len(tr_s_preds)]) + accuracy_score(tr_e_preds, trYe[:len(tr_e_preds)]))/2
    va_acc = (accuracy_score(va_s_preds, vaYs[:len(va_s_preds)]) + accuracy_score(va_e_preds, vaYe[:len(va_e_preds)]))/2
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f'%(n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            save(os.path.join(save_dir, desc, 'best_params.jl'))

def predict():
    s_preds, e_preds = iter_predict(teCtxW, teQW, teCtxCh, teQCh)
    path = os.path.join(submission_dir, desc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\t{}\t{}\t{}\n'.format('index', 'start prediction', 'start target', 'end prediction', 'end target'))
        for i, (s_pred, s_targ, e_pred, e_targ) in enumerate(zip(s_preds, teYs, e_preds, teYe)):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(i, s_pred, int(s_targ), e_pred, int(e_targ)))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='qanet') # dir args
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--encoding_dir', type=str, default='data/glove_vocab/')
    parser.add_argument('--data_dir', type=str, default='data/squad_1.1/')
    parser.add_argument('--glove_dir', type=str, default='data/pretrained_glove_vectors/')
    parser.add_argument('--use_prev_best', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--data_limit', type=int)
    parser.add_argument('--seed', type=int, default=42) # seed
    parser.add_argument('--n_gpu', type=int, default=1) # train args
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--max_ctx', type=int, default=512) # model params
    parser.add_argument('--max_q', type=int, default=128)
    parser.add_argument('--char_dim', type=int, default=16)
    parser.add_argument('--max_words', type=int, default=200000)
    parser.add_argument('--n_wembd', type=int, default=300)
    parser.add_argument('--n_cembd', type=int, default=200)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=int, default=1) # opt args
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # log args
    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    # handle data 
    (trCtxW, trQW, trCtxCh, trQCh, trYs, trYe), (
        vaCtxW, vaQW, vaCtxCh, vaQCh, vaYs, vaYe), (teCtxW, teQW, teCtxCh, teQCh, teYs, teYe), config = squad_data(max_ctx, max_q, encoding_dir, data_dir, char_dim, max_words, data_limit=data_limit)
    globals().update(config)
    n_train = len(trCtxW)
    n_valid = len(vaCtxW)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter
    # place holders
    Cw_train = tf.placeholder(tf.int32, [n_batch_train, max_ctx])
    Qw_train = tf.placeholder(tf.int32, [n_batch_train, max_q])
    Cch_train = tf.placeholder(tf.int32, [n_batch, max_ctx, char_dim])
    Qch_train = tf.placeholder(tf.int32, [n_batch, max_q, char_dim])
    Ys_train = tf.placeholder(tf.int32, [n_batch_train])
    Ye_train = tf.placeholder(tf.int32, [n_batch_train])
    Cw = tf.placeholder(tf.int32, [n_batch_train, max_ctx])
    Qw = tf.placeholder(tf.int32, [n_batch_train, max_q])
    Cch = tf.placeholder(tf.int32, [n_batch, max_ctx, char_dim])
    Qch = tf.placeholder(tf.int32, [n_batch, max_q, char_dim])
    Ys = tf.placeholder(tf.int32, [n_batch_train])
    Ye = tf.placeholder(tf.int32, [n_batch_train])
    # mgpu train and predict
    train, s_preds, e_preds, qa_losses = mgpu_train(Cw_train, Qw_train, Cch_train, Qch_train, Ys_train, Ye_train)
    qa_loss = tf.reduce_mean(qa_losses)
    eval_mgpu_s_preds, eval_mgpu_e_preds, eval_mgpu_qa_losses = mgpu_predict(Cw_train, Qw_train, Cch_train, Qch_train, Ys_train, Ye_train)
    eval_s_preds, eval_e_preds, eval_qa_losses = model(Cw, Qw, Cch, Qch, Ys, Ye,
        n_word=n_word, n_char=n_char, n_pred=n_pred, n_wembd=n_wembd, n_cembd=n_cembd, units=units, embd_pdrop=embd_pdrop, n_head=n_head, 
        attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, train=False, reuse=True)
    eval_mgpu_qa_loss = tf.reduce_mean(eval_mgpu_qa_losses)
    eval_qa_loss = tf.reduce_mean(eval_qa_losses)
    # params
    params = find_trainable_variables('model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    # get saved params
    sess.run([tf.global_variables()[1].assign(np.load('{}glove_300_400k_matrix.npy'.format(glove_dir))[:n_word])])
    if use_prev_best and os.path.isfile(os.path.join(save_dir, desc, 'best_params.jl')):
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
    # train, eval, test
    n_updates = 0
    n_epochs = 0
    if submit:
        save(os.path.join(save_dir, desc, 'best_params.jl'))
    best_score = 0
    for i in range(n_iter):
        for cwmb, qwmb, cchmb, qchmb, ysmb, yemb in iter_data(*shuffle(trCtxW, trQW, trCtxCh, trQCh, trYs, trYe, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            cost, _ = sess.run([qa_loss, train], {Cw_train:cwmb, Qw_train:qwmb, Cch_train:cchmb, Qch_train:qchmb, Ys_train:ysmb, Ye_train:yemb})
            n_updates += 1
            if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                log()
        n_epochs += 1
        log()
    if submit:
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
        predict()