"""cbows model and skip-gram model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.liner1 = nn.Linear(context_size*embedding_dim, 128)
        self.liner2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)  
        out = F.relu(self.liner1(embeds))
        out = self.liner2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs

    #Output parameters in output file
    def output_vec(self, id2word, output_file_name):
        """Save embedding parameters into output file
        """
        embeddings = self.embeddings.weight.data.numpy()
        with open(output_file_name, 'w') as file:
            file.write("Length of the dictionary: %d | Embedding dimension %d\n\n" % (len(id2word), self.embedding_dim))
            for wid, word in id2word.items():
                file.write("%s    " % word)
                file.write("%s\n\n" % embeddings[wid])

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.liner1 = nn.Linear(embedding_dim, 128)
        self.liner2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)  
        out = F.relu(self.liner1(embeds))
        out = self.liner2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs

    #Output parameters in output file
    def output_vec(self, id2word, output_file_name):
        """Save embedding parameters into output file
        """
        embeddings = self.embeddings.weight.data.numpy()
        with open(output_file_name, 'w') as file:
            file.write("Length of the dictionary: %d" % len(id2word), "| Embedding dimension: %d\n" % self.embedding_dim)
            for wid, word in id2word.items():
                file.write("%s" % word)
                file.write("%s" % embeddings[wid])

        
# model = CBOW(3,2,2)
# print(model)
