import tensorflow as tf
import numpy as np
import os, time, math, json, joblib, random, argparse

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from util.opt import adam
from util.utils import iter_data, find_trainable_variables, ResultLogger, assign_to_gpu, average_grads, make_path
from data.data import snli_data

from models.transformer_clf import clf_model as model

# modified from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            clf_logits, clf_losses, lm_losses = model(*xs, 
                clf_token=clf_token, n_vocab=n_vocab, n_class=n_class, n_special=n_special, n_ctx=n_ctx, 
                n_embd=n_embd, embd_pdrop=embd_pdrop, n_layer=n_layer, n_head=n_head, attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop, clf_pdrop=clf_pdrop, train=True, reuse=do_reuse)
            if lm_coef > 0:
                train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
            else:
                train_loss = tf.reduce_mean(clf_losses)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
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
            clf_logits, clf_losses, lm_losses = model(*xs, 
                clf_token=clf_token, n_vocab=n_vocab, n_class=n_class, n_special=n_special, n_ctx=n_ctx, 
                n_embd=n_embd, embd_pdrop=embd_pdrop, n_layer=n_layer, n_head=n_head, attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop, clf_pdrop=clf_pdrop, train=False, reuse=True)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def iter_apply(Xs, Ms, Ys):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_logits, eval_mgpu_clf_loss], {X_train:xmb, M_train:mmb, Y_train:ymb})
        else:
            res = sess.run([eval_logits, eval_clf_loss], {X:xmb, M:mmb, Y:ymb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=True, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits

def save(path):
    ps = sess.run(params)
    joblib.dump(ps, make_path(path))

def log():
    global best_score
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost/len(trX[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = accuracy_score(np.argmax(tr_logits, 1), trY[:n_valid])
    va_acc = accuracy_score(np.argmax(va_logits, 1), vaY)
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f'%(n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            save(os.path.join(save_dir, desc, 'best_params.jl'))

def predict():
    predictions = np.argmax(iter_predict(teX, teM), 1)
    path = os.path.join(submission_dir, desc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\t{}\n'.format('index', 'prediction', 'target'))
        for i, (prediction, target) in enumerate(zip(predictions, teY)):
            f.write('{}\t{}\t{}\n'.format(i, prediction, int(target)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='transformer_snli') # dir args
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--encoding_dir', type=str, default='data/bpe_encoding/')
    parser.add_argument('--data_dir', type=str, default='data/snli_1.0/')
    parser.add_argument('--pretrained_lm_dir', type=str, default='data/pretrained_language_model_params/')
    parser.add_argument('--use_prev_best', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--data_limit', type=int)
    parser.add_argument('--seed', type=int, default=42) # seed
    parser.add_argument('--n_gpu', type=int, default=1) # train args
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--n_ctx', type=int, default=512) # model params
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--lm_coef', type=float, default=0.5) # opt args
    parser.add_argument('--max_grad_norm', type=int, default=1)
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
    (trX, trM, trY), (vaX, vaM, vaY), (teX, teM, teY), config = snli_data(n_ctx, encoding_dir, data_dir, data_limit=data_limit)
    globals().update(config)
    n_train = len(trX)
    n_valid = len(vaX)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter
    # place holders
    X_train = tf.placeholder(tf.int32, [n_batch_train, n_ctx, 2])
    M_train = tf.placeholder(tf.float32, [n_batch_train, n_ctx])
    Y_train = tf.placeholder(tf.int32, [n_batch_train])
    X = tf.placeholder(tf.int32, [None, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, n_ctx])
    Y = tf.placeholder(tf.int32, [None])
    # mgpu train and predict
    train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train)
    clf_loss = tf.reduce_mean(clf_losses)
    eval_mgpu_logits, eval_mgpu_clf_losses, eval_mgpu_lm_losses = mgpu_predict(X_train, M_train, Y_train)
    eval_logits, eval_clf_losses, eval_lm_losses = model(X, M, Y,
                clf_token=clf_token, n_vocab=n_vocab, n_class=n_class, n_special=n_special, n_ctx=n_ctx, 
                n_embd=n_embd, embd_pdrop=embd_pdrop, n_layer=n_layer, n_head=n_head, attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop, clf_pdrop=clf_pdrop, train=False, reuse=True)
    eval_clf_loss = tf.reduce_mean(eval_lm_losses)
    eval_mgpu_clf_loss = tf.reduce_mean(eval_mgpu_clf_losses)
    # params
    params = find_trainable_variables('model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    # get saved params
    if use_prev_best and os.path.isfile(os.path.join(save_dir, desc, 'best_params.jl')):
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
    else:
        shapes = json.load(open('{}params_shapes.json'.format(pretrained_lm_dir)))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load('{}params_{}.npy'.format(pretrained_lm_dir, n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:n_ctx]
        if n_ctx > 512:
            init_params[0] = np.concatenate([init_params[0], [init_params[0][-1] for i in range(n_ctx-512)]])
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
        del init_params[1]
        sess.run([p.assign(ip) for p, ip in zip(params[:145], init_params[:145])])
    # train, eval, test
    n_updates = 0
    n_epochs = 0
    if submit:
        save(os.path.join(save_dir, desc, 'best_params.jl'))
    best_score = 0
    for i in range(n_iter):
        for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trY, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            cost, _ = sess.run([clf_loss, train], {X_train:xmb, M_train:mmb, Y_train:ymb})
            n_updates += 1
            if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                log()
        n_epochs += 1
        log()
    if submit:
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
        predict()