import nltk, re
from nltk.corpus import gutenberg
from collections import defaultdict,Counter
import utils
import random


def get_sent_tokens(fileid):
    processed_sent_tokens = []
    pattern = re.compile('[\W_]+')
    sents = gutenberg.sents(fileid)
    for s in sents:
        s = (pattern.sub('', w) for w in s)
        s = (w.lower() for w in s if w)
        processed_sent_tokens.append(s)
    return processed_sent_tokens

def get_vocab(fileids):
    vocab = set()
    for f in fileids:
        print (f'processing file: {f} for vocab construction')
        for s in get_sent_tokens(f):
            vocab = vocab.union(set(s))
    vocab = sorted(list(vocab)) + ['UNK']
    return vocab

def get_word_freq_map (fileids):
    word_freq_map = defaultdict(int)
    for f in fileids:
        print(f'processing file: {f} for word freq counting')
        for s in get_sent_tokens(f):
            word_counts = Counter(s)
            for w,c in word_counts.items():
                word_freq_map[w] += c

    word_freq_map['UNK'] =  word_freq_map['the'] #treat it like a stopword
    return word_freq_map


def get_target_context_pairs_sg(fileid, word_to_id_map, max_win_size=3):
    tgt_cont_pairs = []
    for tokens in get_sent_tokens(fileid):
        tokens = list(tokens)
        for i, w in enumerate(tokens):
            win_size = random.randint(1, max_win_size)  # favour closer words
            target = [word_to_id_map[w]]*win_size*2
            context_words = tokens[min(0,i-win_size):i] + tokens[i+1:min(i+win_size+1, len(tokens))]
            context = [word_to_id_map[c] for c in context_words]
            tc_pairs = list(zip(target,context))
            tgt_cont_pairs.extend(tc_pairs)
    return tgt_cont_pairs

def get_filenames_ids():
    fileids = gutenberg.fileids()[:]
    return fileids

def main():
    fileids = gutenberg.fileids()[:10]

    vocab = get_vocab(fileids)
    print (f'len of vocab: {len(vocab)}')

    token_to_id_map, id_to_token_map = utils.get_token_id_maps(vocab)
    word_freq = get_word_freq_map(fileids)

    print (word_freq)

    for f in fileids:
        print (get_target_context_pairs_sg(f, token_to_id_map, win_size=3))
        input()


if __name__ == '__main__':
    pass