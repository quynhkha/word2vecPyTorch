import os, glob, random
import torch
import skipgram, cbow
import numpy as np
from time import time
import utils


def main():
    dataset_folder = '../data/sentiment_labelled_sentences'
    n_epochs = 30
    embedding_size = 32
    num_negsamples = 3
    win_size = 3
    lr = 0.01

    if torch.cuda.is_available():
        device = torch.device("cuda")

    files = glob.glob(dataset_folder+"/*.txt")
    n_files = len(files)
    print (f'loaded {n_files} files from {dataset_folder}')

    vocab = utils.get_vocab(files)
    token_to_id_map, id_to_token_map = utils.get_token_id_maps(vocab)
    word_freq_map = utils.get_word_freq_map(files)
    unigram_table = utils.get_unigram_table(word_freq_map, token_to_id_map)


    model = cbow.cbowns(num_words=len(vocab), embedding_dim=embedding_size)
    if 'cuda' == device.type: model = model.cuda()
    print (f' the model being initialized is: {model}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print (f'optimizer used: {optimizer}')

    for e in range(1, n_epochs+1):
        t0 = time()
        random.shuffle(files)
        losses = []
        for fname in files:
            optimizer.zero_grad()
            # file_tc_pairs = utils.get_target_context_pairs_sg(fname,token_to_id_map, win_size=win_size)
            # tcn_tuples = utils.get_tcn_tuples_sg(file_tc_pairs, unigram_table, num_negsamples=num_negsamples)

            file_tc_pairs = utils.get_target_context_pairs_cbow(fname, token_to_id_map, win_size=win_size)
            tcn_tuples = utils.get_tcn_tuples_cbow(file_tc_pairs, unigram_table, num_negsamples=num_negsamples)
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


if __name__ == '__main__':
    main()