import gensim, numpy as np
from gensim.models import KeyedVectors

words = [l.strip() for l in open('vocab.txt').readlines()]
print (f'loaded {len(words)} words')
emb_fname = 'word_embeddings_.txt'
word_emb = np.loadtxt(emb_fname,dtype='str')
print (f'word embedding shape: {word_emb.shape}')

wwe = list(zip(words, word_emb.tolist()))
lines = []
lines.append(str(word_emb.shape[0])+' '+str(word_emb.shape[1]))
for w, we in wwe:
    l = w + ' ' + ' '.join(we)
    lines.append(l)

with open('word2vec_pre_kv_c', 'w') as fh:
    for l in lines:
        print (l,file=fh)

wv_from_text = KeyedVectors.load_word2vec_format('word2vec_pre_kv_c', binary=False)
print (wv_from_text)