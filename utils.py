import numpy
from collections import deque
from nltk.tokenize import word_tokenize
import nltk
import torch

nltk.download("punkt")

class InputData:
    """Process input data into word2vec format for use.

    Attributes:
        word2id: A map with key: id and value: word
        id2word: A map with key: word and value: id
        vocab: The set of all words
        trigrams: A array with [[2][1]] size to represent the context
    """
    def __init__(self, filename):
        self.filename = filename
        self.rawFile = self.load_data()
        self.fileArray = self.rawFile.split()
        self.vocab = set(self.fileArray)
        self.id2word = {i: word for i, word in enumerate(self.vocab)} # Make a map, key: id, value: word
        self.word2id = {word: i for i, word in enumerate(self.vocab)} # Make a map, key: word, value: id
        self.trigrams = [([self.fileArray[i], self.fileArray[i+1]], self.fileArray[i+2]) for i in range(len(self.fileArray) - 2)]
        self.cbow_train_data = self.cbow_prepare()
        self.skip_gram_data = self.skip_gram_prepare()

    def load_data(self):
        with open(self.filename, "rb") as file:
            processsed_text = file.read().decode("utf-8").strip() #strip is to remove start and end chars in the file
            # print(type(processsed_text))
        return processsed_text

    def cbow_prepare(self):
        data = []
        for i in range(2, len(self.fileArray) - 2):
            context = [self.fileArray[i - 2], self.fileArray[i - 1], self.fileArray[i + 1], self.fileArray[i + 2]]
            target = self.fileArray[i]
            data.append((context, target))
        return data
    
    def skip_gram_prepare(self):
        data = []
        for i in range(2, len(self.fileArray) - 2):
            target = [self.fileArray[i - 2], self.fileArray[i - 1], self.fileArray[i + 1], self.fileArray[i + 2]]
            context = self.fileArray[i]
            data.append((context, target))
        return data
        
        
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)
            
# if __name__ == "__main__":
#     dataset = InputData("medium_text.txt")
    # print(dataset.word2id)
    # print(dataset.id2word)
    # print(dataset.cbow_train_data[:5])


