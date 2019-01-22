import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from utils import *
import model

#Hyper parameters
FILENAME = "medium_text.txt"
EPCHO = 2
LR = 0.001
CONTEXT_SIZE = 4
EMBEDDING_DIM = 10

#Data prepare
data_set = InputData(FILENAME)
train_set = data_set.skip_gram_data[:int(len(data_set.skip_gram_data) * 0.6)]
test_set = data_set.skip_gram_data[int(len(data_set.skip_gram_data) * 0.6):]
vocab = data_set.vocab
word_dict = data_set.word2id

#Train prepare
loss_func = nn.NLLLoss()
model = model.SkipGram(len(vocab), EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=LR)
losses = []

#train
for epoch in range(EPCHO):
    total_loss = 0
    for step, (context, target) in enumerate(train_set):
        context_idxs = torch.tensor([word_dict[context]], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_func(log_probs, torch.tensor([word_dict[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % 100 == 0:
            # test_output = model()
            print("Epoch: %s " % epoch, "| loss: %.4f" % loss.data.numpy())
    losses.append(total_loss)
print(model.embeedings.parameters)

