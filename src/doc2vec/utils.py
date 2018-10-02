
import numpy as np


def dump_doc_names(id_to_doc_map):
    min_id = min(id_to_doc_map.keys())
    max_id = max(id_to_doc_map.keys())
    with open('docs.txt', 'w') as fh:
        for id in range(min_id, max_id+1):
            doc_name = id_to_doc_map[id]
            print (doc_name, file=fh)


def get_doc_id_maps(doc_start_neuron_id, doc_fnames):
    doc_fnames = sorted(doc_fnames)
    doc_to_id_map = {w: i+doc_start_neuron_id for i, w in enumerate(doc_fnames)}
    id_to_doc_map = {i: w for w, i in doc_to_id_map.items()}
    return doc_to_id_map, id_to_doc_map

def get_doc_id_maps_split(doc_fnames):
    doc_fnames = sorted(doc_fnames)
    doc_to_id_map = {w: i for i, w in enumerate(doc_fnames)}
    id_to_doc_map = {i: w for w, i in doc_to_id_map.items()}
    return doc_to_id_map, id_to_doc_map


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
    unigram_table = np.array(unigram_table, dtype=np.int64)
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
    try:
        t,c = zip(*file_tc_pairs)
    except:
        t = []
        c = []
        n = []
    tcn_pairs = list(zip(t,c,n))
    return tcn_pairs


def get_batch(tcn_tuples, batch_size):
    for batch_start_index in range(0, len(tcn_tuples), batch_size):
        yield tcn_tuples[batch_start_index:batch_start_index + batch_size]


def dump_vocab(id_to_token_map):
    with open('vocab.txt', 'w') as fh:
        max_id = max(id_to_token_map)
        for w_id in range(max_id + 1):
            print(id_to_token_map[w_id], file=fh)



if __name__ == '__main__':
    pass