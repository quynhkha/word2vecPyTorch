import os, glob, random
import torch
import skipgram
import numpy as np
from time import time
import utils
# from gutenberg_corpus_parser import *
from sentiment_corpus_parser import *


def main():
    n_epochs = 50
    embedding_size = 300
    num_negsamples = 5
    win_size = 5
    lr = 0.005
    batch_size = 1024

    if torch.cuda.is_available():
        device = torch.device("cuda")

    files = get_filenames_ids('../../data/sentiment_labelled_sentences/imdb_doc_dataset/')
    files = [f for f in files if '_labels.txt' not in f]
    print (f'considering {len(files)} files for doc embedding')
    n_docs = len(files)

    vocab = get_vocab(files)
    n_words = len(vocab)
    doc_start_neuron_id = n_words

    token_to_id_map, id_to_token_map = utils.get_token_id_maps(vocab)
    doc_to_id_map, id_to_doc_map = utils.get_doc_id_maps(doc_start_neuron_id, files)
    word_freq_map = get_word_freq_map(files)
    unigram_table = utils.get_unigram_table(word_freq_map, token_to_id_map)

    utils.dump_vocab(id_to_token_map)
    utils.dump_doc_names(id_to_doc_map)


    model = skipgram.sgns(num_words=n_words, num_docs = n_docs, embedding_dim=embedding_size)
    if 'cuda' == device.type: model = model.cuda()
    print (f' the model being initialized is: {model}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print (f'optimizer used: {optimizer}')

    for e in range(1, n_epochs+1):
        t0 = time()
        random.shuffle(files)
        losses = []
        for fname in files:
            # print ('training ', fname)
            word_tc_pairs, doc_tc_pairs = get_target_context_pairs_sg(fname, token_to_id_map, doc_to_id_map, max_win_size=win_size)
            word_tcn_tuples = utils.get_tcn_tuples_sg(word_tc_pairs, unigram_table, num_negsamples=num_negsamples)
            doc_tcn_tuples = utils.get_tcn_tuples_sg(doc_tc_pairs, unigram_table, num_negsamples=num_negsamples)
            tcn_tuples = word_tcn_tuples + doc_tcn_tuples
            random.shuffle(tcn_tuples)
            for batch_num, batch_tcn in enumerate(utils.get_batch(tcn_tuples, batch_size)):
                t, c, n = zip(*batch_tcn)
                loss = model.forward(t, c, n, device=device, num_negsample=num_negsamples)
                losses.append(loss.data[0][0])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (batch_num+1) %5 == 0:
                    print (f"epoch: {e}, file: {fname}, batch: {batch_num+1}, loss: {loss.data[0][0]}")

        epoch_loss = sum(losses)/len(losses)
        epoch_time = time() - t0
        print(f'loss : {epoch_loss}, epoch: {e}, time: {round(epoch_time, 4)} sec')

        if e%2 == 0:
            word_embeddings = model.get_embeddings(type='word')
            doc_embeddings = model.get_embeddings(type='doc')
            np.savetxt('word_embeddings_{}.txt'.format(e),word_embeddings)
            np.savetxt('doc_embeddings_{}.txt'.format(e),doc_embeddings)

    print ('end')


if __name__ == '__main__':
    main()