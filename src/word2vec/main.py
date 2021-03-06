import os, glob, random
import torch
import skipgram, cbow
import numpy as np
from time import time
import utils
from gutenberg_corpus_parser import *
# from sentiment_corpus_parser import *


def main():
    n_epochs = 30
    embedding_size = 96
    num_negsamples = 5
    win_size = 5
    lr = 0.005
    batch_size = 10240

    if torch.cuda.is_available():
        device = torch.device("cuda")

    files = get_filenames_ids()

    vocab = get_vocab(files)
    token_to_id_map, id_to_token_map = utils.get_token_id_maps(vocab)
    word_freq_map = get_word_freq_map(files)
    unigram_table = utils.get_unigram_table(word_freq_map, token_to_id_map)

    utils.dump_vocab(id_to_token_map)

    model = skipgram.sgns(num_words=len(vocab), embedding_dim=embedding_size)
    # model = cbow.cbowns(num_words=len(vocab), embedding_dim=embedding_size)
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

            file_tc_pairs = get_target_context_pairs_sg(fname,token_to_id_map, max_win_size=win_size)
            tcn_tuples = utils.get_tcn_tuples_sg(file_tc_pairs, unigram_table, num_negsamples=num_negsamples)

            # file_tc_pairs = get_target_context_pairs_cbow(fname, token_to_id_map, win_size=win_size)
            # tcn_tuples = utils.get_tcn_tuples_cbow(file_tc_pairs, unigram_table, num_negsamples=num_negsamples)


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
            target_word_embeddings = model.get_embeddings()
            np.savetxt('word_embeddings_{}.txt'.format(e),target_word_embeddings)


    print ('end')


if __name__ == '__main__':
    main()