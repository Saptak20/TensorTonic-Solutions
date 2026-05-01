import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)
    
    result = []
    
    for seq in seqs:
        if len(seq) >= max_len:
            result.append(seq[:max_len])
        else:
            result.append(seq + [pad_value] * (max_len - len(seq)))
    
    return np.array(result)