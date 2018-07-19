import numpy as np
import pandas as pd
import json, itertools, ftfy, spacy, re
from tqdm import tqdm
from util.text_utils import TextEncoder
from util.utils import encode_dataset

def snli_data(max_len, encoding_dir, data_dir, data_limit=None):
    # paths
    encoder_path = encoding_dir + "encoder_bpe_40000.json"
    vocab_path = encoding_dir + "vocab_40000.bpe"
    train_path = data_dir + "snli_1.0_train.txt"
    val_path = data_dir + "snli_1.0_dev.txt"
    test_path = data_dir + "snli_1.0_test.txt"
    # encoder
    text_encoder = TextEncoder(encoder_path, vocab_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    encoder['<s>'] = len(encoder)
    start = encoder['<s>']
    encoder['<d>'] = len(encoder)
    delimiter = encoder['<d>']
    encoder['<e>'] = len(encoder)
    end = encoder['<e>']
    n_special = len(encoder) - n_vocab
    # read csv's
    tr = pd.concat(pd.read_csv(train_path, chunksize=100000, sep='\t', 
        usecols=['gold_label', 'sentence1','sentence2'], keep_default_na=False), ignore_index=True)
    va = pd.read_csv(val_path, sep='\t', usecols=['gold_label', 'sentence1','sentence2'], keep_default_na=False)
    te = pd.read_csv(test_path, sep='\t', usecols=['gold_label', 'sentence1','sentence2'], keep_default_na=False)
    trY, trX1, trX2 = list(tr[tr.columns[0]]), list(tr[tr.columns[1]]), list(tr[tr.columns[2]])
    vaY, vaX1, vaX2 = list(va[va.columns[0]]), list(va[va.columns[1]]), list(va[va.columns[2]])
    teY, teX1, teX2 = list(te[te.columns[0]]), list(te[te.columns[1]]), list(te[te.columns[2]])
    # encode data
    if data_limit:
        trX1, trX2, vaX1, vaX2, teX1, teX2 = trX1[:data_limit], trX2[:data_limit], vaX1[:data_limit], vaX2[:data_limit], teX1[:data_limit], teX2[:data_limit]
    (trX1, trX2), (vaX1, vaX2), (teX1, teX2) = encode_dataset(
        [(trX1, trX2), (vaX1, vaX2), (teX1, teX2)], encoder=text_encoder)
    max_len = max_len//2+(-n_special//2)
    n_ctx = n_special+max(max([len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(trX1, trX2)]),
               max([len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(vaX1, vaX2)]),
               max([len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(teX1, teX2)]))
    # pack data into tensors
    def transform_snli(X1, X2, Y):
        n_batch = len(X1)
        xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
        ymb = np.zeros((n_batch), dtype=np.float32)
        for i, (x1, x2, y) in tqdm(enumerate(zip(X1, X2, Y)), ncols=80, leave=False):
            x = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[end]
            l = len(x)
            xmb[i, :l, 0] = x
            mmb[i, :l] = 1
            ymb[i] = 0 if y == "neutral" else (1 if y == "entailment" else 2)
        xmb[:, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
        return xmb, mmb, ymb
    trX, trM, trY = transform_snli(trX1, trX2, trY)
    vaX, vaM, vaY = transform_snli(vaX1, vaX2, vaY)
    teX, teM, teY = transform_snli(teX1, teX2, teY)
    # return config dict
    def get_dict(**args):
        return args
    config = get_dict(n_vocab=n_vocab, n_special=n_special, n_ctx=n_ctx, clf_token=end, n_class=3)
    return (trX, trM, trY), (vaX, vaM, vaY), (teX, teM, teY), config

def wikitext_data(max_len, encoding_dir, data_dir, min_chars=80, tokens=False, data_limit=None):
    # paths
    ext = "tokens" if tokens else "raw"
    encoder_path = encoding_dir + "encoder_bpe_40000.json"
    vocab_path = encoding_dir + "vocab_40000.bpe"
    train_path = data_dir + "/wiki.train.raw"
    val_path = data_dir + "/wiki.valid.raw"
    test_path = data_dir + "/wiki.test.raw"
    # encoder
    text_encoder = TextEncoder(encoder_path, vocab_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    encoder['<s>'] = len(encoder)
    start = encoder['<s>']
    encoder['<e>'] = len(encoder)
    end = encoder['<e>']
    decoder = {v:k for k,v in encoder.items()}
    n_special = len(encoder) - n_vocab
    # read csv's
    trX = pd.concat(pd.read_csv(train_path, chunksize=100000, sep='\n', header=None), ignore_index=True)
    vaX = pd.read_csv(val_path, sep='\n', header=None)
    teX = pd.read_csv(test_path, sep='\n', header=None)
    trX = list(trX[trX[0].map(len) > min_chars][0])
    vaX = list(vaX[vaX[0].map(len) > min_chars][0])
    teX = list(teX[teX[0].map(len) > min_chars][0])
    n_ctx = max(max([len(x[:max_len]) for x in trX]),
               max([len(x[:max_len]) for x in vaX]),
               max([len(x[:max_len]) for x in teX]))
    # encode data
    if data_limit:
        trX, vaX, teX = trX[:data_limit], vaX[:data_limit], teX[:data_limit]
    (trX,), (vaX,), (teX,) = encode_dataset([(trX,), (vaX,), (teX,)], encoder=text_encoder)
    # pack data into tensors
    def transform_wikitext(X):
        n_batch = len(X)
        xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
        for i, x in tqdm(enumerate(X), ncols=80, leave=False):
            x = [start]+x[:n_ctx-2]+[end]
            l = len(x)
            xmb[i, :l, 0] = x
            mmb[i, :l] = 1
        xmb[:, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
        return xmb, mmb
    trX, trM = transform_wikitext(trX)
    vaX, vaM = transform_wikitext(vaX)
    teX, teM = transform_wikitext(teX)
    # return config dict
    def get_dict(**args):
        return args
    config = get_dict(n_vocab=n_vocab, n_special=n_special, n_ctx=n_ctx, decoder=decoder)
    return (trX, trM), (vaX, vaM), (teX, teM), config

def gen_squad_data(max_len, encoding_dir, data_dir, data_limit=None):
    # paths
    encoder_path = encoding_dir + "encoder_bpe_40000.json"
    vocab_path = encoding_dir + "vocab_40000.bpe"
    train_path = data_dir + "train-v1.1.json"
    val_path = data_dir + "dev-v1.1.json"
    test_path = data_dir + "dev-v1.1.json"
    # encoder
    text_encoder = TextEncoder(encoder_path, vocab_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    encoder['<s>'] = len(encoder)
    start = encoder['<s>']
    encoder['<as>'] = len(encoder)
    ans_start = encoder['<as>']
    encoder['<ae>'] = len(encoder)
    ans_end = encoder['<ae>']
    encoder['<d>'] = len(encoder)
    delimiter = encoder['<d>']
    encoder['<e>'] = len(encoder)
    end = encoder['<e>']
    decoder = {v:k for k,v in encoder.items()}
    n_special = len(encoder) - n_vocab
    # split data
    def get_split_set(squad_set):
        pre, answers, post, questions = [], [], [], []
        for block in squad_set['data']:
            for para in block['paragraphs']:
                for qa in para['qas']:
                    ans_set = set([(ans['text'], ans['answer_start']) for ans in qa['answers']])
                    for ans in ans_set:
                        pre.append(para['context'][:ans[1]])
                        answers.append(para['context'][ans[1]:ans[1]+len(ans[0])])
                        post.append(para['context'][ans[1]+len(ans[0]):])
                        questions.append(qa['question'])
        return pre, answers, post, questions
    trPre, trAns, trPost, trQ = get_split_set(json.load(open(train_path)))
    vaPre, vaAns, vaPost, vaQ = get_split_set(json.load(open(val_path)))
    tePre, teAns, tePost, teQ = get_split_set(json.load(open(test_path)))
    # encode data
    if data_limit:
        trPre, trAns, trPost, trQ, vaPre, vaAns, vaPost, vaQ, tePre, teAns, tePost, teQ = (
            trPre[:data_limit], trAns[:data_limit], trPost[:data_limit], trQ[:data_limit], vaPre[:data_limit], vaAns[:data_limit], 
            vaPost[:data_limit], vaQ[:data_limit], tePre[:data_limit], teAns[:data_limit], tePost[:data_limit], teQ[:data_limit])
    (trPre, trAns, trPost, trQ), (vaPre, vaAns, vaPost, vaQ), (tePre, teAns, tePost, teQ
        ) = encode_dataset([(trPre, trAns, trPost, trQ), (vaPre, vaAns, vaPost, vaQ
        ), (tePre, teAns, tePost, teQ)], encoder=text_encoder)
    n_ctx=max_len
    # pack data into tensors
    def transform_gen_squad(X1, X2, X3, X4):
        n_batch = len(X1)
        xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
        minus = 0
        for i, (x1, x2, x3, x4) in tqdm(enumerate(zip(X1, X2, X3, X4)), ncols=80, leave=False):
            x = [start]+x1+[ans_start]+x2+[ans_end]+x3+[delimiter]+x4+[end]
            l = len(x)
            if l <= n_ctx:
                xmb[i-minus, :l, 0] = x
                mmb[i-minus, :l] = 1
            else:
                minus+=1
        if minus > 0:
            xmb, mmb = xmb[:-minus], mmb[:-minus]
        xmb[:, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
        return xmb, mmb
    trX, trM = transform_gen_squad(trPre, trAns, trPost, trQ)
    vaX, vaM = transform_gen_squad(vaPre, vaAns, vaPost, vaQ)
    teX, teM = transform_gen_squad(tePre, teAns, tePost, teQ)
    # return config dict
    def get_dict(**args):
        return args
    config = get_dict(n_vocab=n_vocab, n_special=n_special, n_ctx=n_ctx, decoder=decoder, delimiter=delimiter, end=end)
    return (trX, trM), (vaX, vaM), (teX, teM), config

def squad_data(max_ctx, max_q, encoding_dir, data_dir, char_dim=16, max_words=200000, data_limit=None):
    # paths
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
    vocab_path = encoding_dir + "vocab.txt"
    train_path = data_dir + "train-v1.1.json"
    val_path = data_dir + "dev-v1.1.json"
    test_path = data_dir + "dev-v1.1.json"
    # split data
    def get_split_set(squad_set):
        contexts, questions, answers = [], [], []
        for block in squad_set['data']:
            for para in block['paragraphs']:
                for qa in para['qas']:
                    if (len(para['context'].split()) < max_ctx and 
                            len(qa['question'].split()) < max_q):
                        contexts.append(para['context'])
                        questions.append(qa['question'])
                        answers.append(qa['answers'][0])
        return contexts, questions, answers
    trCtx, trQ, trAns = get_split_set(json.load(open(train_path)))
    vaCtx, vaQ, vaAns = get_split_set(json.load(open(val_path)))
    teCtx, teQ, teAns = get_split_set(json.load(open(test_path)))
    # encode data
    word_dict = dict()
    word_dict['<unk>'] = len(word_dict)
    for line in open(vocab_path).read().split("\n")[:max_words]:
        word_dict[line.split()[1]] = len(word_dict)
    char_dict = ['<nan>']+sorted(set("".join(list(itertools.chain(*[
        trCtx, trQ, vaCtx, vaQ, teCtx, teQ])))))
    char_dict = {k: v for v, k in enumerate(char_dict)}

    def convert_idx(text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token.text, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token.text)))
            current += len(token.text)
        return spans

    decoder = {v:k for k,v in word_dict.items()}

    def encode_set(CTX, Q, ANS):
        ctxW, ctxCh, yS, yE = encode(CTX, ANS)
        qW, qCh = encode(Q)
        return ctxW, qW, ctxCh, qCh, yS, yE
    
    def encode(texts, answers=None):
        seqW = []
        seqCh = []
        if answers:
            yS = []
            yE = []
        for i, text_str in tqdm(enumerate(texts), ncols=80, leave=False):
            text_str = text_standardize(ftfy.fix_text(text_str))
            text = nlp(text_str)
            if answers:
                spans = convert_idx(text_str, text)
                answer_start = answers[i]['answer_start']
                answer_end = answer_start + len(answers[i]['text'])
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start > span[1]):
                        start_idx = max(idx-1,0)
                        break
                si, ei = _find_sub_tokens(text[start_idx:], text_standardize(ftfy.fix_text(answers[i]['text'])))
                yS.append(start_idx+si)
                yE.append(start_idx+ei)
            word_tokens = []
            char_tokens = []
            for token in text:
                t = token.text
                word_tokens.append(word_dict.get(t.lower(), 0))
                char_tokens.append([char_dict.get(t[i], 0) if i < len(t) else 0 for i in range(char_dim)])
            seqW.append(word_tokens)
            seqCh.append(char_tokens)
        if answers:
            return seqW, seqCh, yS, yE
        return seqW, seqCh
    if data_limit:
        trCtx, trQ, trAns, vaCtx, vaQ, vaAns, teCtx, teQ, teAns = (
            trCtx[:data_limit], trQ[:data_limit], trAns[:data_limit], vaCtx[:data_limit], vaQ[:data_limit], 
            vaAns[:data_limit], teCtx[:data_limit], teQ[:data_limit], teAns[:data_limit])
    trCtxW, trQW, trCtxCh, trQCh, trYs, trYe = encode_set(trCtx, trQ, trAns)
    vaCtxW, vaQW, vaCtxCh, vaQCh, vaYs, vaYe = encode_set(vaCtx, vaQ, vaAns)
    teCtxW, teQW, teCtxCh, teQCh, teYs, teYe = encode_set(teCtx, teQ, teAns)
    # pack data into tensors
    def transform_squad(CTXW, QW, CTXCH, QCH, YS, YE):
        n_batch = len(CTXW)
        cwmb = np.zeros((n_batch, max_ctx), dtype=np.int32)
        qwmb = np.zeros((n_batch, max_q), dtype=np.int32)
        cchmb = np.zeros((n_batch, max_ctx, char_dim), dtype=np.int32)
        qchmb = np.zeros((n_batch, max_q, char_dim), dtype=np.int32)
        ysmb = np.zeros((n_batch), dtype=np.int32)
        yemb = np.zeros((n_batch), dtype=np.int32)
        minus = 0
        for i, (ctxW, qW, ctxCh, qCh, yS, yE) in tqdm(enumerate(zip(CTXW, QW, CTXCH, QCH, YS, YE)), ncols=80, leave=False):
            if len(ctxW) <= max_ctx and len(qW) <= max_q and len(ctxCh) <= max_ctx and len(qCh) <= max_q:
                cwmb[i-minus, :len(ctxW)] = ctxW
                qwmb[i-minus, :len(qW)] = qW
                cchmb[i-minus, :len(ctxCh), :] = ctxCh
                qchmb[i-minus, :len(qCh), :] = qCh
                ysmb[i-minus] = yS
                yemb[i-minus] = yE
            else:
                minus+=1
        if minus > 0:
            cwmb, qwmb, cchmb, qchmb, ysmb, yemb = (
                cwmb[:-minus], qwmb[:-minus], cchmb[:-minus], qchmb[:-minus], ysmb[:-minus], yemb[:-minus])
        return cwmb, qwmb, cchmb, qchmb, ysmb, yemb
    trCtxW, trQW, trCtxCh, trQCh, trYs, trYe = transform_squad(trCtxW, trQW, trCtxCh, trQCh, trYs, trYe)
    vaCtxW, vaQW, vaCtxCh, vaQCh, vaYs, vaYe = transform_squad(vaCtxW, vaQW, vaCtxCh, vaQCh, vaYs, vaYe)
    teCtxW, teQW, teCtxCh, teQCh, teYs, teYe = transform_squad(teCtxW, teQW, teCtxCh, teQCh, teYs, teYe)
    # return config dict
    def get_dict(**args):
        return args
    config = get_dict(n_word=len(word_dict), n_char=len(char_dict), max_q=max_q, max_ctx=max_ctx)
    return (trCtxW, trQW, trCtxCh, trQCh, trYs, trYe), (
        vaCtxW, vaQW, vaCtxCh, vaQCh, vaYs, vaYe), (teCtxW, teQW, teCtxCh, teQCh, teYs, teYe), config

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return text.strip()

def _find_sub_tokens(tokens, ans):
    infer_len = min(len(ans.split()), len(tokens))
    while infer_len <= len(tokens):
        if ans in tokens[:infer_len].text:
            return _get_idxs(tokens[:infer_len], ans)
        infer_len += 1
    print("Sequence {} cannot be found".format(ans))
    raise Exception()

def _get_idxs(tokens, ans):
    si = 0
    ei = len(tokens)
    while ei-1 > si and ans in tokens[:ei-1].text:
        ei-=1
    while si+1 < ei and ans in tokens[si+1:].text:
        si+=1
    return si, ei