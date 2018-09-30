
import numpy as np


def get_token_id_maps(vocab):
    token_to_id_map = {w:i for i,w in enumerate(vocab)}
    id_to_token_map = {i:w for w,i in token_to_id_map.items()}
    return token_to_id_map, id_to_token_map


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


def get_tcn_tuples_sg(file_tc_pairs, unigram_table, num_negsamples):
    # tcn_pairs = []
    # for t, c in file_tc_pairs:
    #     n = np.random.choice(unigram_table, num_negsamples)
    #     while c in n:
    #         n = np.random.choice(unigram_table, num_negsamples)
    #     tcn = (t,c,n.tolist())
    #     tcn_pairs.append(tcn)

    n = np.random.choice(unigram_table, size=(len(file_tc_pairs), num_negsamples))
    t,c = zip(*file_tc_pairs)
    tcn_pairs = list(zip(t,c,n))
    return tcn_pairs




if __name__ == '__main__':
    pass