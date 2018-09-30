import torch
import torch.nn as nn
import torch.nn.functional as F


class cbowns(nn.Module):
    def __init__(self, num_words, embedding_dim):
        super(cbowns, self).__init__()
        self.embedding_size = embedding_dim
        self.context_emb = nn.Embedding(num_words, embedding_dim)  # hidden layer embeddings
        self.target_emb = nn.Embedding(num_words, embedding_dim)  # output layer embeddings

        init_range = 0.005
        self.context_emb.weight.data.uniform_(-init_range, init_range)
        self.target_emb.weight.data.zero_()

    # def _get_variables_for_embedding_lookup(self, targets, contexts, negsamples, num_negsample, device):
    #     if 'cuda' == device.type:
    #         targets = torch.cuda.LongTensor(targets, device).view(-1, 1)
    #         contexts = torch.cuda.LongTensor(contexts, device)#.view(-1, 1)
    #         negsamples = torch.cuda.LongTensor(negsamples, device)#.view(-1, num_negsample)
    #     else:
    #         targets = torch.LongTensor(targets, device).view(-1, 1)
    #         contexts = torch.LongTensor(contexts, device)#.view(-1, 1)
    #         negsamples = torch.LongTensor(negsamples, device)#.view(-1, num_negsample)
    #     return targets, contexts, negsamples
    #
    # def forward(self, targets, contexts, negsamples, device, num_negsample=3):
    #     targets, contexts, negsamples = self._get_variables_for_embedding_lookup(targets, contexts, negsamples,
    #                                                                              num_negsample, device)
    #     target_embeds = self.target_emb(targets)  # B x 1 x D
    #     context_embeds = self.context_emb(contexts)  # B x 1 x D
    #     neg_embeds = -self.context_emb(negsamples)  # B x K x D
    #
    #     context_embeds = context_embeds.mean(1).view(-1,1,self.embedding_size)
    #     neg_embeds = neg_embeds.mean(1).view(-1,1,self.embedding_size)
    #
    #     positive_score = context_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2)  # Bx1
    #     negative_score = neg_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2)
    #     # loss1 = F.logsigmoid(torch.clamp(positive1_score,max=10,min=-10)) + F.logsigmoid(torch.clamp(negative1_score,max=10,min=-10))
    #     loss = F.logsigmoid(positive_score) + F.logsigmoid(negative_score)
    #     loss = -torch.mean(loss)
    #
    #     return loss


    def _get_variables_for_embedding_lookup(self, targets, contexts, negsamples, num_negsample, device):
        if 'cuda' == device.type:
            targets = torch.cuda.LongTensor(targets).view(-1, 1)
            contexts = torch.cuda.LongTensor(contexts).view(-1, 1)
            negsamples = torch.cuda.LongTensor(negsamples).view(-1, num_negsample)
        else:
            targets = torch.LongTensor(targets).view(-1, 1)
            contexts = torch.LongTensor(contexts).view(-1, 1)
            negsamples = torch.LongTensor(negsamples).view(-1, num_negsample)
        return targets, contexts, negsamples


    def forward(self, targets, contexts, negsamples, device, num_negsample=3):
        losses = torch.tensor([0], device=device, requires_grad=True, dtype=torch.float32)
        tcn = list(zip(targets, contexts, negsamples))
        for t,c,n in tcn:
            targets, contexts, negsamples = self._get_variables_for_embedding_lookup(t, c, n, num_negsample, device)
            target_embeds = self.target_emb(targets)  # B x 1 x D
            context_embeds = self.context_emb(contexts)  # B x 1 x D
            neg_embeds = -self.context_emb(negsamples)  # B x K x D
            try:
                context_embeds = context_embeds.mean(0).view(-1,1,self.embedding_size)
            except:
                print ('err case')
                continue
            neg_embeds = neg_embeds.mean(0).view(-1,num_negsample,self.embedding_size)

            positive_score = context_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2)  # Bx1
            negative_score = torch.sum(neg_embeds.bmm(target_embeds.transpose(1, 2)).squeeze(2), 1)#.view(negsamples.size(0),-1)
            # loss1 = F.logsigmoid(torch.clamp(positive1_score,max=10,min=-10)) + F.logsigmoid(torch.clamp(negative1_score,max=10,min=-10))
            loss = F.logsigmoid(positive_score) + F.logsigmoid(negative_score)
            losses = losses.add(loss)

        loss = -losses/len(tcn)
        return loss

    def get_embeddings(self):
        return self.target_emb.weight.data.cpu().numpy()