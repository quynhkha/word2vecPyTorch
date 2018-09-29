from collections import defaultdict,Counter
from nltk.tokenize import RegexpTokenizer
import numpy as np


def get_vocab(files):
    vocab = set()
    for f in files:
        lines = open(f).readlines()
        for l in lines:
            l = l.lower()
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = set(tokenizer.tokenize(l))
            vocab = vocab.union(tokens)
    vocab = sorted(list(vocab)) + ['UNK']
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

    word_freq_map['UNK'] =  word_freq_map['the'] #treat it like a stopword
    return word_freq_map


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


def get_target_context_pairs_sg(fname, word_to_id_map, win_size=3):
    tgt_cont_pairs = []
    lines = open(fname).readlines()
    for l in lines:
        l = l.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(l)
        for i, w in enumerate(tokens):
            target = [word_to_id_map[w]]*win_size*2
            context_words = tokens[min(0,i-win_size):i] + tokens[i+1:min(i+win_size+1, len(tokens))]
            context = [word_to_id_map[c] for c in context_words]
            tc_pairs = list(zip(target,context))
            tgt_cont_pairs.extend(tc_pairs)
    return tgt_cont_pairs


def get_tcn_tuples_sg(file_tc_pairs, unigram_table, num_negsamples):
    tcn_pairs = []
    for t, c in file_tc_pairs:
        n = np.random.choice(unigram_table, num_negsamples)
        while c in n:
            n = np.random.choice(unigram_table, num_negsamples)
        tcn = (t,c,n.tolist())
        tcn_pairs.append(tcn)
    return tcn_pairs



def get_target_context_pairs_cbow(fname, word_to_id_map, win_size=3):
    tgt_cont_pairs = []
    lines = open(fname).readlines()
    for l in lines:
        l = l.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(l)
        for i, w in enumerate(tokens):
            target = word_to_id_map[w]
            context_words = tokens[max(0,i-win_size):i] + tokens[i+1:min(i+win_size+1, len(tokens))]
            context = [word_to_id_map[c] for c in context_words]
            context = context + [word_to_id_map['UNK']] * (2*win_size-len(context))
            tc_pairs = (target,context)
            tgt_cont_pairs.append(tc_pairs)
    return tgt_cont_pairs


def get_tcn_tuples_cbow(file_tc_pairs, unigram_table, num_negsamples):
    tcn_pairs = []
    for t, c in file_tc_pairs:
        n = np.random.choice(unigram_table, num_negsamples*len(c))
        #known issue: the intersection of c and n is checked for all values of c. Ideally is should be different for each item in c?
        # while set(c).intersection(set(n)):
        #     n = np.random.choice(unigram_table, num_negsamples*len(c))
        tcn = (t,c,n.tolist())
        tcn_pairs.append(tcn)
    return tcn_pairs


if __name__ == '__main__':
    pass