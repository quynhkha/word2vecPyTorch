import torch
import torch.nn as nn
import torch.nn.functional as F

class sgns(nn.Module):

    def __init__(self, num_words, embedding_dim):
        super(sgns,self).__init__()
        self.target_emb = nn.Embedding(num_words, embedding_dim) # hidden layer embeddings
        self.context_emb = nn.Embedding(num_words, embedding_dim) # output layer embeddings

        init_range = 0.005
        self.target_emb.weight.data.uniform_(-init_range, init_range)
        self.context_emb.weight.data.zero_()

    def _get_variables_for_embedding_lookup(self, targets, contexts, negsamples, num_negsample, device):
        if 'cuda' == device.type:
            targets = torch.cuda.LongTensor(targets, device).view(-1, 1)
            contexts = torch.cuda.LongTensor(contexts, device).view(-1, 1)
            negsamples = torch.cuda.LongTensor(negsamples, device).view(-1, num_negsample)
        else:
            targets = torch.LongTensor(targets, device).view(-1, 1)
            contexts = torch.LongTensor(contexts, device).view(-1, 1)
            negsamples = torch.LongTensor(negsamples, device).view(-1, num_negsample)
        return targets, contexts, negsamples

    def forward_sg(self, targets, contexts, negsamples, device, num_negsample=3):
        targets, contexts, negsamples = self._get_variables_for_embedding_lookup(targets, contexts, negsamples,
                                                                                 num_negsample, device)
        target_embeds = self.target_emb(targets)  # B x 1 x D
        context_embeds = self.context_emb(contexts)  # B x 1 x D
        neg_embeds = -self.context_emb(negsamples)  # B x K x D

        positive_score = context_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2)  # Bx1
        negative_score = torch.sum(neg_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2), 1).view(negsamples.size(0),-1)  # BxK -> Bx1
        # loss1 = F.logsigmoid(torch.clamp(positive1_score,max=10,min=-10)) + F.logsigmoid(torch.clamp(negative1_score,max=10,min=-10))
        loss = F.logsigmoid(positive_score) + F.logsigmoid(negative_score)
        loss = -torch.mean(loss)

        return loss

    def get_embeddings(self):
        return self.target_emb.weight.data.cpu().numpy()