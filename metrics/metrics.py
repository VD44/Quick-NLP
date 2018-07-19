import numpy as np
from metrics.rouge import rouge

def np_rouge(val, ref, start, end):
    def trim_seq(seq, start, end):
        seq = seq[list(seq).index(start)+1:] if start in seq else seq
        seq = seq[:list(seq).index(end)] if end in seq else seq
        return np.trim_zeros(seq,'b')
    val, ref = list(val), list(ref)
    for i in range(len(val)):
        val[i] = " ".join(str(c) for c in trim_seq(val[i], start, end))
        ref[i] = " ".join(str(c) for c in trim_seq(ref[i], start, end))
    return rouge(val, ref)

def target_based_np_rouge(val, ref, start, end, tz=True):
    def trim_seqs(val, ref, start, end):
        start_idx = list(ref).index(start)+1 if start in ref else 0
        val = val[start_idx:]
        ref = ref[start_idx:]
        val = val[:list(val).index(end)] if end in val else val
        ref = ref[:list(ref).index(end)] if end in ref else ref
        if tz:
            val = np.trim_zeros(val,'b')
            ref = np.trim_zeros(ref,'b')
        return val, ref
    val, ref = list(val), list(ref)
    for i in range(len(val)):
        sval, sref = trim_seqs(val[i], ref[i], start, end)
        sval = " ".join(str(c) for c in sval)
        sref = " ".join(str(c) for c in sref)
        val[i] = sval
        ref[i] = sref
    return rouge(val, ref)