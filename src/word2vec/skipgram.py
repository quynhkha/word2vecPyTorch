import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class sgns(nn.Module):

    def __init__(self, num_words, embedding_dim):
        super(sgns,self).__init__()
        self.embedding_size = embedding_dim
        self.target_emb = nn.Embedding(num_words, embedding_dim) # hidden layer embeddings
        self.context_emb = nn.Embedding(num_words, embedding_dim) # output layer embeddings

        init_range = 0.005
        self.target_emb.weight.data.uniform_(-init_range, init_range)
        self.context_emb.weight.data.zero_()

    def _get_tensor_for_embedding_lookup(self, targets, contexts, negsamples, device):

        targets = torch.tensor(targets,dtype=torch.long,device=device).view(-1, 1)
        contexts = torch.tensor(contexts,dtype=torch.long,device=device).view(-1, 1)
        negsamples = torch.tensor(negsamples,dtype=torch.long,device=device).view(-1, 1)

        return targets, contexts, negsamples

    def forward(self, targets, contexts, negsamples, device, num_negsample=3):
        targets, contexts, negsamples = self._get_tensor_for_embedding_lookup(targets, contexts, negsamples, device)
        target_embeds = self.target_emb(targets)  # B x 1 x D
        context_embeds = self.context_emb(contexts)  # B x 1 x D
        neg_embeds = -self.context_emb(negsamples)
        neg_embeds = neg_embeds.view(-1,num_negsample,self.embedding_size)# B x K x D

        positive_score = context_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2)  # Bx1
        negative_score = neg_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2)
        negative_score = torch.sum(negative_score, 1).view(-1,1)
        # loss1 = F.logsigmoid(torch.clamp(positive1_score,max=10,min=-10)) + F.logsigmoid(torch.clamp(negative1_score,max=10,min=-10))
        loss = F.logsigmoid(positive_score) + F.logsigmoid(negative_score)
        loss = -torch.mean(loss)

        return loss

    def get_embeddings(self):
        return self.target_emb.weight.data.cpu().numpy()