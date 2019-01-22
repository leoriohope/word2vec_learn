"""cbows model and skip-gram model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class Cbow(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Cbow, self).__init__()
        self.embeedings = nn.Embedding(vocab_size, embedding_dim)
        self.liner1 = nn.Linear(context_size*embedding_dim, 128)
        self.liner2 = nn.Linear(128, vocab_size)
        


