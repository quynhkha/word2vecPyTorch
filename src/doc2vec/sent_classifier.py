import numpy as np,os
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
def main():
    for e in range(2,51,2):
        emb_fname = 'doc_embeddings_{}.txt'.format(e)
        labels_fname = '../../data/sentiment_labelled_sentences/imdb_doc_dataset/imdb_labels.txt'
        labels = [int(l.strip()) for l in open(labels_fname)]

        doc_emb_loaded = np.loadtxt(emb_fname)
        doc_fnames = [l.strip() for l in open('docs.txt')]
        doc_embs = list(range(len(doc_emb_loaded)))
        for fname in doc_fnames:
            fname = os.path.basename(fname)
            doc_id = int(fname.split('_')[1].split('.')[0]) - 1
            doc_emb = doc_emb_loaded[doc_id,:]
            doc_embs[doc_id] = doc_emb



        acc = []
        for run in range (1,6):
            XY = list(zip(doc_embs, labels))
            shuffle(XY)
            X,Y = zip(*XY)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
            a = accuracy_score(y_test, y_hat)
            acc.append(a)

        mean_acc = sum(acc)/len(acc)
        print (f'fname: {emb_fname}, acc: {mean_acc}')










if __name__ == '__main__':
    main()