"""cbows model and skip-gram model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeedings = nn.Embedding(vocab_size, embedding_dim)
        self.liner1 = nn.Linear(context_size*embedding_dim, 128)
        self.liner2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeedings(inputs).view(1, -1)  
        out = F.relu(self.liner1(embeds))
        out = self.liner2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeedings = nn.Embedding(vocab_size, embedding_dim)
        self.liner1 = nn.Linear(embedding_dim, 128)
        self.liner2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeedings(inputs).view(1, -1)  
        out = F.relu(self.liner1(embeds))
        out = self.liner2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs
        
# model = CBOW(3,2,2)
# print(model)
