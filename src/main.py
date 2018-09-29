import os, glob, nltk, random
import torch
from pprint import pprint
from collections import defaultdict,Counter
import skipgram
from nltk.tokenize import RegexpTokenizer
import numpy as np
from time import time

def get_vocab(files):
    vocab = set()
    for f in files:
        lines = open(f).readlines()
        for l in lines:
            l = l.lower()
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = set(tokenizer.tokenize(l))
            vocab = vocab.union(tokens)
    vocab = sorted(list(vocab))
    return vocab


def get_token_id_maps(vocab):
    token_to_id_map = {w:i for i,w in enumerate(vocab)}
    id_to_token_map = {i:w for w,i in token_to_id_map.items()}
    return token_to_id_map, id_to_token_map


def get_word_freq_map (files):
    word_freq_map = defaultdict(int)
    for f in files:
        lines = open(f).readlines()
        for l in lines:
            l = l.lower()
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(l)
            word_counts = Counter(tokens)
            for w,c in word_counts.items():
                word_freq_map[w] += c
    return word_freq_map


def get_target_context_pairs(fname, word_to_id_map, win_size=3):
    tgt_cont_pairs = []
    lines = open(fname).readlines()
    for l in lines:
        l = l.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(l)
        for i, w in enumerate(tokens):
            target = [word_to_id_map[w]]*win_size*2
            #known bug = if i-win <0, tokens[i-win_size:i] is totally ignored
            context_words = tokens[i-win_size:i] + tokens[i+1:i+win_size+1]
            context = [word_to_id_map[c] for c in context_words]
            tc_pairs = list(zip(target,context))
            tgt_cont_pairs.extend(tc_pairs)
    return tgt_cont_pairs


def get_unigram_table(word_freq_map, token_to_id_map):
    total_word_count = sum(word_freq_map.values())
    word_probs = {w: word_freq_map[w] /total_word_count for w in word_freq_map.keys()}
    unigram_table = []
    unigram_table_size = 2000
    raised_word_probs = {w: p ** 0.75 for w, p in word_probs.items()}
    word_to_negsample_freq = {w: p * unigram_table_size for w, p in raised_word_probs.items()}

    for w, f in word_to_negsample_freq.items():
        unigram_table.extend([token_to_id_map[w]] * (int(f) + 1))
    unigram_table = np.array(unigram_table, dtype=np.int16)
    return unigram_table


def get_tcn_tuples(file_tc_pairs, unigram_table, num_negsamples):
    tcn_pairs = []
    for t, c in file_tc_pairs:
        n = np.random.choice(unigram_table, num_negsamples)
        while c in n:
            n = np.random.choice(unigram_table, num_negsamples)
        tcn = (t,c,n.tolist())
        tcn_pairs.append(tcn)
    return tcn_pairs

dataset_folder = '../data/sentiment_labelled_sentences'
n_epochs = 40
embedding_size = 32
num_negsamples = 3
win_size = 3
lr = 0.05

if torch.cuda.is_available():
    device = torch.device("cuda")


files = glob.glob(dataset_folder+"/*.txt")
n_files = len(files)
print (f'loaded {n_files} files from {dataset_folder}')

vocab = get_vocab(files)
token_to_id_map, id_to_token_map = get_token_id_maps(vocab)
word_freq_map = get_word_freq_map(files)
unigram_table = get_unigram_table(word_freq_map, token_to_id_map)


model = skipgram.sgns(num_words=len(vocab), embedding_dim=embedding_size)
if torch.cuda.is_available():
    model = model.cuda()
print (model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for e in range(1, n_epochs+1):
    t0 = time()
    random.shuffle(files)
    losses = []
    for fname in files:
        optimizer.zero_grad()
        file_tc_pairs = get_target_context_pairs(fname,token_to_id_map, win_size=win_size)
        tcn_tuples = get_tcn_tuples(file_tc_pairs, unigram_table, num_negsamples=num_negsamples)
        random.shuffle(tcn_tuples)

        t, c, n = zip(*tcn_tuples)

        loss = model.forward(t, c, n, device=device, num_negsample=num_negsamples)

        losses.append(loss.data[0])
        loss.backward()

        optimizer.step()

    epoch_loss = sum(losses)/len(losses)
    epoch_time = time() - t0
    print(f'loss : {epoch_loss}, epoch: {e}, time: {epoch_time}')


target_word_embeddings = model.get_embeddings()

with open('vocab.txt','w') as fh:
    max_id = max(id_to_token_map)
    for w_id in range(max_id+1):
        print(id_to_token_map[w_id], file=fh)

np.savetxt('word_embeddings.txt',target_word_embeddings, fmt='%.8f')


print ('end')