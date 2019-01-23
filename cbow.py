import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from utils import *
import model

#Hyper parameters
FILENAME = "medium_text.txt"
EPCHO = 500
LR = 0.001
CONTEXT_SIZE = 4
EMBEDDING_DIM = 5
OUTFILE = "out_parameters.txt"

#Data prepare
data_set = InputData(FILENAME)
train_set = data_set.cbow_train_data[:int(len(data_set.cbow_train_data) * 0.8)]
test_set = data_set.cbow_train_data[int(len(data_set.cbow_train_data) * 0.8):]
vocab = data_set.vocab
word_dict = data_set.word2id
id2word = data_set.id2word

#Train prepare
loss_func = nn.NLLLoss()
model = model.CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LR)
losses = []

#train
for epoch in range(EPCHO):
    total_loss = 0
    for step, (context, target) in enumerate(train_set):
        context_idxs = make_context_vector(context, word_dict)
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

print(losses)
model.output_vec(id2word, OUTFILE)
